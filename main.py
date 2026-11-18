#!/usr/bin/env python3
"""BTC Race v5, Variant B — trend-following bot with a trailing stop.

Trades cbBTC/USDC (CBBTC) minute by minute on agent-fun/bars-replay/v1:
reads NDJSON on stdin (hello, one batch per minute, done) and answers every
batch with exactly one reply carrying its orders.

Strategy (no dip-buying, no mean reversion — pure trend-following):
  * Warmup: trade nothing until 30 bars of history (or the harness warmup,
    whichever is longer) while EMAs build.
  * Entry: buy 50% of equity when the EMA(8) of closes crosses above
    EMA(21) AND the close sits above the 60-bar mean (confirmed uptrend),
    AND the cross is decisive (EMA(8) above EMA(21) by at least 1 bp),
    AND at least 60 bars have passed since the last exit, AND the last
    10 bars' range is not extreme. If a buy that was sent is rejected
    (gap-up, no bar), the entry is retried for up to 5 bars while the
    setup still holds, instead of the cross being lost forever. A cross
    that the filters blocked is NOT retried: the setup itself was not
    valid then. One position at a time, never adds, never buys while a
    buy is still awaiting its fill.
  * Exit: trailing stop — sell everything when the close falls at least
    TRAIL_BPS below the highest close since entry. Hard stop-loss
    HARD_STOP_BPS below the fill price, anchored to the lower of the fill
    price and the fill bar's close so a gap-up fill cannot whipsaw the
    stop on the next bar. No take-profit and no exit on an EMA cross
    back: on the dev window the cross-back exit churned every winner.
    A sell rejected for max_slippage is resent with escalating slippage
    (300 -> 1000 -> 10000 bps) so a crash can never lock the bot in.
  * Safety: no entries after bar 1400 (a 1-bar hold is pure cost);
    no entries while equity sits more than 2% below its peak (circuit
    breaker — exits always stay allowed).
  * Filter: skip entries while the last 10 bars' range is extreme
    (top-decile 1-minute volatility on the dev window), to avoid whipsaw
    chop.

Risk: one 50%-of-equity position. When fills land near the stop, a
single losing trade costs about 0.25% of equity plus ~26 bps round-trip
cost; in a gapping market the loss can be larger, which the escalating
sell slippage and the circuit breaker bound.

Deterministic: no clock, no network, no files, no unseeded randomness.
Every decision is a pure function of the bars seen so far, so two runs on
the same data produce byte-identical action logs.

Standard library only.
"""

import json
import sys

# --- Tunables (tuned on the frozen dev window; robust set, not the peak) ---
EMA_FAST = 8            # fast EMA length in bars
EMA_SLOW = 21           # slow EMA length in bars
MEAN_WIN = 60           # bars in the trend-confirmation mean
WARMUP_BARS = 30        # bars of history before the first trade
ENTRY_FRAC_BPS = 5000   # buy size: 50% of equity
TRAIL_BPS = 80          # trailing stop: 0.8% below the peak close since entry
HARD_STOP_BPS = 50      # hard stop: 0.5% below the (anchored) fill price
VOL_WIN = 10            # bars in the volatility-filter range
VOL_MAX_BPS = 46        # skip entries when the 10-bar range tops this (dev p90)
MAX_IMPACT_BPS = 10     # cap a buy so its own price impact stays under this
BUY_SLIPPAGE_BPS = 300  # max move from the latest close a buy fill may take
SELL_SLIPPAGE_STEPS = (300, 1000, 10000)  # escalate on rejected sells
ENTRY_RETRY_BARS = 5    # bars a cross stays armed after a rejected buy
ENTRY_CUTOFF_BAR = 1400  # no new entries after this many bars seen
CIRCUIT_BREAKER_BPS = 200  # block entries while equity is this far below peak
COOLDOWN_BARS = 60      # bars to wait after an exit before a fresh entry
MIN_SEP_BPS = 1         # entry needs EMA(8) above EMA(21) by at least this
MEAN_RISING_BARS = 0    # 0 = off; the rising-mean filter cost a winner on dev
EXIT_ON_CROSS = False   # no exit on EMA cross back: it churned every winner

SYMBOL = "CBBTC"
BPS = 10_000


def ema_update(prev, price, length):
    """One EMA step. prev is None on the first bar (seed with the price)."""
    if prev is None:
        return float(price)
    k = 2.0 / (length + 1)
    return price * k + prev * (1.0 - k)


class TrendBot:
    """Keeps the bars, the EMAs and the open-trade state."""

    def __init__(self, hello):
        self.hello = hello
        sym = next(s for s in hello["symbols"] if s["symbol"] == SYMBOL)
        self.price_scale = sym["price_scale"]
        self.qty_scale = sym["qty_scale"]
        self.quote_scale = hello["capital"]["quote_scale"]
        self.cash = int(hello["capital"]["quote_units"])
        self.equity = self.cash
        fees = hello.get("fees", {})
        self.min_notional = int(
            fees.get("min_notional_quote_units", 1_000_000)
        )
        # Trade only once both our own warmup and the harness warmup are over.
        self.trade_from_bar = max(WARMUP_BARS, int(hello.get("warmup_bars", 0)))
        self.closes = []          # int closes, latest last
        self.highs = []           # int highs, latest last
        self.lows = []            # int lows, latest last
        self.depths = []          # latest close depth lists, latest last
        self.ema_fast = None
        self.ema_slow = None
        self.prev_fast = None     # EMA values as of the previous bar
        self.prev_slow = None
        self.position_qty = 0     # qty held (raw units at qty_scale)
        self.entry_price = None   # anchored fill price of the open trade (raw)
        self.peak_close = None    # highest close since entry (raw)
        self.means = []           # 60-bar mean after each bar (for the rising check)
        self.bars_since_exit = 10 ** 9  # bars since the last sell fill
        self.buy_rejected = False  # a sent buy was rejected; retry is armed
        self.retry_deadline = None  # bar count by which a retry must fire
        self.awaiting_buy = False  # a buy was sent, its fill not yet reported
        self.pending_sell = False  # a sell was rejected; resend it
        self.sell_rejects = 0     # consecutive rejected sells (drives slippage)
        self.peak_equity = self.cash
        self.next_id = 1

    def _client_id(self):
        cid = "b%d" % self.next_id
        self.next_id += 1
        return cid

    def _mean_close(self):
        window = self.closes[-MEAN_WIN:]
        return sum(window) / len(window)

    def _range_bps(self):
        """Range of the last VOL_WIN bars in bps of the low."""
        window_h = self.highs[-VOL_WIN:]
        window_l = self.lows[-VOL_WIN:]
        low = min(window_l)
        if low <= 0:
            return 0
        return (max(window_h) - low) * BPS // low

    def _impact_cap(self, depth):
        """Largest buy notional (quote units) whose impact stays under
        MAX_IMPACT_BPS on the given pool depth (quote units per hop)."""
        if not depth:
            return 0
        cap = None
        for d in depth:
            room = MAX_IMPACT_BPS * d // BPS
            cap = room if cap is None else min(cap, room)
        return cap if cap is not None else 0

    def _sell_order(self):
        """The exit order, with slippage escalated per consecutive reject."""
        step = min(self.sell_rejects, len(SELL_SLIPPAGE_STEPS) - 1)
        return {
            "side": "sell",
            "client_id": self._client_id(),
            "symbol": SYMBOL,
            "qty": str(self.position_qty),
            "max_slippage_bps": SELL_SLIPPAGE_STEPS[step],
        }

    def on_batch(self, batch):
        """Decide the orders for one batch. Returns a list of wire orders."""
        # Fold the account in first: cash, equity and the held position.
        account = batch["account"]
        self.cash = int(account["cash_quote_units"])
        self.equity = int(account["equity_quote_units"])
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        positions = account.get("positions", [])
        self.position_qty = int(positions[0]["qty"]) if positions else 0

        # Fold the new bar in (CBBTC may be absent on a gap minute).
        bar = None
        for b in batch.get("bars", []):
            if b["symbol"] == SYMBOL:
                bar = b
                break
        if bar is not None:
            close = int(bar["c"])
            self.closes.append(close)
            self.highs.append(int(bar["h"]))
            self.lows.append(int(bar["l"]))
            self.depths.append([int(d) for d in bar["c_depth"]])
            self.prev_fast = self.ema_fast
            self.prev_slow = self.ema_slow
            self.ema_fast = ema_update(self.ema_fast, close, EMA_FAST)
            self.ema_slow = ema_update(self.ema_slow, close, EMA_SLOW)
            self.means.append(self._mean_close())
            self.bars_since_exit += 1

        # Fold the fills in: a filled buy opens the trade, a filled sell
        # closes it, a rejected sell arms the escalating resend.
        latest_close = self.closes[-1] if self.closes else None
        for fill in batch.get("fills", []):
            if fill["side"] == "buy":
                self.awaiting_buy = False
                if fill["status"] == "filled":
                    price = int(fill["price"])
                    # Anchor the hard stop to the lower of the fill price
                    # and the fill bar's close: a gap-up fill must not
                    # whipsaw the stop on the next bar.
                    if latest_close is not None and latest_close < price:
                        price = latest_close
                    self.entry_price = price
                    self.peak_close = price
                    self.pending_sell = False
                    self.sell_rejects = 0
                    self.buy_rejected = False
                    self.retry_deadline = None
                elif fill["status"] == "rejected":
                    # Only a buy the bot actually sent arms the retry, and
                    # the deadline runs from the FIRST rejection of the
                    # setup: repeated rejections do not extend it.
                    if self.retry_deadline is None:
                        # The rejection batch itself counts as retry bar
                        # one, so the window ends ENTRY_RETRY_BARS - 1
                        # bars later: at most 5 retry attempts in all.
                        self.retry_deadline = (
                            len(self.closes) + ENTRY_RETRY_BARS - 1
                        )
                    self.buy_rejected = True
            elif fill["side"] == "sell":
                if fill["status"] == "filled":
                    self.entry_price = None
                    self.peak_close = None
                    self.pending_sell = False
                    self.sell_rejects = 0
                    self.bars_since_exit = 0
                elif fill["status"] == "rejected":
                    self.pending_sell = True
                    self.sell_rejects += 1

        # --- Exit logic (holding a position; runs even on gap batches) ---
        if self.position_qty > 0 and self.closes:
            close = self.closes[-1]
            if self.peak_close is None:
                self.peak_close = close
            elif close > self.peak_close:
                self.peak_close = close
            trail_hit = close * BPS <= self.peak_close * (BPS - TRAIL_BPS)
            stop_hit = (
                self.entry_price is not None
                and close * BPS <= self.entry_price * (BPS - HARD_STOP_BPS)
            )
            cross_down = (
                EXIT_ON_CROSS
                and self.prev_fast is not None
                and self.prev_slow is not None
                and self.prev_fast >= self.prev_slow
                and self.ema_fast < self.ema_slow
            )
            if self.pending_sell or trail_hit or stop_hit or cross_down:
                return [self._sell_order()]
            return []

        # --- Entry logic (flat; needs a fresh bar) ---
        if bar is None:
            return []
        if len(self.closes) <= self.trade_from_bar:
            return []
        if len(self.closes) < MEAN_WIN:
            return []  # the 60-bar mean must be a full window, not partial
        if len(self.closes) > ENTRY_CUTOFF_BAR:
            return []  # too late in the window: a new hold is pure cost
        if self.awaiting_buy:
            return []  # a buy is already in flight; never double it
        # Circuit breaker: stop entering while equity is far below its peak.
        # Exits always stay allowed (handled above).
        if self.equity * BPS < self.peak_equity * (BPS - CIRCUIT_BREAKER_BPS):
            return []
        crossed_up = (
            self.prev_fast is not None
            and self.prev_slow is not None
            and self.prev_fast <= self.prev_slow
            and self.ema_fast > self.ema_slow
        )
        retry_armed = (
            self.buy_rejected
            and self.retry_deadline is not None
            and len(self.closes) <= self.retry_deadline
            and self.ema_fast > self.ema_slow
        )
        if crossed_up:
            # A fresh cross starts a fresh setup: any stale retry deadline
            # from an earlier cross does not carry over.
            self.retry_deadline = None
        if not (crossed_up or retry_armed):
            return []
        # The cross must be decisive, not a hairline touch of two flat EMAs.
        if (self.ema_fast - self.ema_slow) * BPS < MIN_SEP_BPS * close:
            return []
        if close <= self._mean_close():
            return []  # not a confirmed uptrend: close below the 60-bar mean
        # The trend itself must be turning up, not just the price above a
        # falling average (disabled: it cost a winner on dev).
        if (
            MEAN_RISING_BARS > 0
            and len(self.means) > MEAN_RISING_BARS
            and self.means[-1] <= self.means[-1 - MEAN_RISING_BARS]
        ):
            return []
        if self.bars_since_exit < COOLDOWN_BARS:
            return []  # just stopped out of the same chop: wait for a new setup
        # A retry re-attempts an entry the bot already approved when it
        # sent the rejected buy, so the volatility filter (which judged
        # the setup at the cross) does not apply to it again: a gap-up
        # rejection inflates the 10-bar range for 10 bars, which would
        # otherwise block every retry bar.
        if not retry_armed and self._range_bps() > VOL_MAX_BPS:
            return []  # chop: the last 10 bars swung too wide to trust a cross
        spend = self.equity * ENTRY_FRAC_BPS // BPS
        spend = min(spend, self.cash)
        depth_cap = self._impact_cap(self.depths[-1])
        if depth_cap > 0:
            spend = min(spend, depth_cap)
        if spend < self.min_notional:
            return []
        self.awaiting_buy = True
        self.buy_rejected = False  # this attempt is in flight; a reject re-arms it
        return [
            {
                "side": "buy",
                "client_id": self._client_id(),
                "symbol": SYMBOL,
                "notional_quote_units": str(spend),
                "max_slippage_bps": BUY_SLIPPAGE_BPS,
            }
        ]


def main():
    bot = None
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue  # malformed line: skip it rather than dying
        kind = msg.get("type") if isinstance(msg, dict) else None
        try:
            if kind == "hello":
                bot = TrendBot(msg)
            elif kind == "batch":
                if bot is None:
                    raise ValueError("batch before hello")
                orders = bot.on_batch(msg)
                out.write(
                    json.dumps(
                        {"type": "reply", "seq": msg["seq"], "orders": orders},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                out.flush()
            elif kind == "done":
                break
        except Exception:
            # Never die mid-run on one bad batch: answer it empty so the
            # harness can carry on, and keep the state as it stands.
            if kind == "batch" and isinstance(msg, dict) and "seq" in msg:
                try:
                    out.write(
                        json.dumps(
                            {"type": "reply", "seq": msg["seq"], "orders": []},
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    out.flush()
                except Exception:
                    break


if __name__ == "__main__":
    main()
