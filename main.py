#!/usr/bin/env python3
"""BTC Race v5, Variant B — trend-following bot with a trailing stop.

Trades cbBTC/USDC (CBBTC) minute by minute on agent-fun/bars-replay/v1:
reads NDJSON on stdin (hello, one batch per minute, done) and answers every
batch with exactly one reply carrying its orders.

Strategy (no dip-buying, no mean reversion — pure trend-following):
  * Warmup: trade nothing for the first 30 bars while EMAs build.
  * Entry: buy 50% of equity when the EMA(8) of closes crosses above
    EMA(21) AND the close sits above the 60-bar mean (confirmed uptrend),
    AND the cross is decisive (EMA(8) above EMA(21) by at least 1 bp),
    AND at least 60 bars have passed since the last exit, AND the last
    10 bars' range is not extreme. One position at a time, never adds.
  * Exit: trailing stop — sell everything when the close falls at least
    TRAIL_BPS below the highest close since entry. Hard stop-loss
    HARD_STOP_BPS below the fill price. No take-profit and no exit on an
    EMA cross back: on the dev window the cross-back exit churned every
    winner (10 trades, -$5.87); the trailing stop alone banks the trend.
    The rising-60-bar-mean filter was tried and dropped: it removed a
    winning trade on dev.
  * Filter: skip entries while the last 10 bars' range is extreme
    (top-decile 1-minute volatility on the dev window), to avoid whipsaw
    chop. Re-entry needs a fresh cross plus the cooldown.

Risk: one 50%-of-equity position, hard stop caps a losing trade near
0.5% of the fill price, so a single loss costs about 0.25% of equity.

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
HARD_STOP_BPS = 50      # hard stop: 0.5% below the fill price
VOL_WIN = 10            # bars in the volatility-filter range
VOL_MAX_BPS = 46        # skip entries when the 10-bar range tops this (dev p90)
MAX_IMPACT_BPS = 10     # cap a buy so its own price impact stays under this
BUY_SLIPPAGE_BPS = 200  # max move from the latest close a buy fill may take
SELL_SLIPPAGE_BPS = 300  # max move from the latest close a sell fill may take
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
        self.cash = int(hello["capital"]["quote_units"])
        self.equity = self.cash
        self.price_scale = hello["symbols"][0]["price_scale"]
        self.qty_scale = hello["symbols"][0]["qty_scale"]
        self.quote_scale = hello["capital"]["quote_scale"]
        self.closes = []          # int closes, latest last
        self.highs = []           # int highs, latest last
        self.lows = []            # int lows, latest last
        self.depths = []          # latest close depth lists, latest last
        self.ema_fast = None
        self.ema_slow = None
        self.prev_fast = None     # EMA values as of the previous bar
        self.prev_slow = None
        self.position_qty = 0     # qty held (raw units at qty_scale)
        self.entry_price = None   # fill price of the open trade (raw)
        self.peak_close = None    # highest close since entry (raw)
        self.means = []           # 60-bar mean after each bar (for the rising check)
        self.bars_since_exit = 10 ** 9  # bars since the last sell fill
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
        MAX_IMPACT_BPS on the given pool depth (USDC raw units per hop)."""
        if not depth:
            return 0
        cap = None
        for d in depth:
            # impact bps of amount a on depth d is ~ a / d * BPS (one hop).
            # a in quote units; depth in USDC raw units: convert.
            room = MAX_IMPACT_BPS * d // BPS
            room = room * (10 ** self.quote_scale) // 1_000_000
            cap = room if cap is None else min(cap, room)
        return cap if cap is not None else 0

    def _qty_value(self, qty, price):
        """What qty is worth in quote units at price (both raw)."""
        return qty * price * (10 ** self.quote_scale) // (
            10 ** (self.qty_scale + self.price_scale)
        )

    def on_batch(self, batch):
        """Decide the orders for one batch. Returns a list of wire orders."""
        # Fold the account in first: cash, equity and the held position.
        account = batch["account"]
        self.cash = int(account["cash_quote_units"])
        self.equity = int(account["equity_quote_units"])
        positions = account.get("positions", [])
        self.position_qty = int(positions[0]["qty"]) if positions else 0

        # A filled buy in this batch's fills opens the trade: record its price.
        for fill in batch.get("fills", []):
            if fill["status"] == "filled" and fill["side"] == "buy":
                self.entry_price = int(fill["price"])
                self.peak_close = self.entry_price
            elif fill["status"] == "filled" and fill["side"] == "sell":
                self.entry_price = None
                self.peak_close = None
                self.bars_since_exit = 0

        # Fold the new bar in (CBBTC may be absent on a gap minute).
        bar = None
        for b in batch.get("bars", []):
            if b["symbol"] == SYMBOL:
                bar = b
                break
        if bar is None:
            return []
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

        if len(self.closes) <= WARMUP_BARS:
            return []

        # --- Exit logic (holding a position) ---
        if self.position_qty > 0:
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
            if trail_hit or stop_hit or cross_down:
                return [
                    {
                        "side": "sell",
                        "client_id": self._client_id(),
                        "symbol": SYMBOL,
                        "qty": str(self.position_qty),
                        "max_slippage_bps": SELL_SLIPPAGE_BPS,
                    }
                ]
            return []

        # --- Entry logic (flat) ---
        if self.prev_fast is None or self.prev_slow is None:
            return []
        crossed_up = self.prev_fast <= self.prev_slow and self.ema_fast > self.ema_slow
        if not crossed_up:
            return []
        # The cross must be decisive, not a hairline touch of two flat EMAs.
        if (self.ema_fast - self.ema_slow) * BPS < MIN_SEP_BPS * close:
            return []
        if close <= self._mean_close():
            return []  # not a confirmed uptrend: close below the 60-bar mean
        # The trend itself must be turning up, not just the price above a
        # falling average: the 60-bar mean must sit above its value
        # MEAN_RISING_BARS ago.
        if (
            MEAN_RISING_BARS > 0
            and len(self.means) > MEAN_RISING_BARS
            and self.means[-1] <= self.means[-1 - MEAN_RISING_BARS]
        ):
            return []
        if self.bars_since_exit < COOLDOWN_BARS:
            return []  # just stopped out of the same chop: wait for a new setup
        if self._range_bps() > VOL_MAX_BPS:
            return []  # chop: the last 10 bars swung too wide to trust a cross
        spend = self.equity * ENTRY_FRAC_BPS // BPS
        spend = min(spend, self.cash)
        depth_cap = self._impact_cap(self.depths[-1])
        if depth_cap > 0:
            spend = min(spend, depth_cap)
        if spend < 1_000_000:  # below the 1 USDC minimum notional
            return []
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
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "hello":
            bot = TrendBot(msg)
        elif kind == "batch":
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


if __name__ == "__main__":
    main()
