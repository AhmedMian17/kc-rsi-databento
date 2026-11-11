"""KC + RSI Mean Reversion: a bar-by-bar port of the Pine Script v6 strategy.

The engine replicates the parts of TradingView's broker emulator that this
strategy relies on:

* Signals are evaluated on each bar's close.
* strategy.entry() is a market order, so it fills at the NEXT bar's open.
* strategy.exit() is called on every bar in the Pine script, so the stop and
  take-profit are recalculated from the latest close every bar and apply to
  the following bar. (See `fixed_stop` below for why this matters.)
* When a bar touches both the stop and the target, TradingView assumes the
  price path O -> H -> L -> C if the high is closer to the open, otherwise
  O -> L -> H -> C. Same here.
* If the open gaps through a stop or target, the fill happens at the open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class Params:
    # Pine inputs (defaults match the script)
    kc_length: int = 20
    kc_mult: float = 2.0
    rsi_length: int = 14
    rsi_ob: float = 70
    rsi_os: float = 30
    adx_length: int = 14
    adx_limit: float = 25
    adx_tf: str | None = None  # None = chart timeframe; or e.g. "60min", "4h"
    risk_reward: float = 1.0

    # Variant switch. False = faithful to the Pine script (stop recalculated
    # every bar from the latest close). True = stop locked in at the signal bar.
    fixed_stop: bool = False

    # Contract and cost settings (MGC: 10 troy oz, 0.10 tick = $1)
    point_value: float = 10.0
    tick_size: float = 0.10
    qty: int = 1
    commission_per_side: float = 0.0  # USD per contract per side
    slippage_ticks: int = 0  # applied to market entries and stop fills
    initial_capital: float = 10_000.0


@dataclass
class Trade:
    direction: int  # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl: float = 0.0
    bars_held: int = 0


@dataclass
class Result:
    params: Params
    bars: pd.DataFrame  # price + indicators + signals
    trades: list[Trade] = field(default_factory=list)

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


# --------------------------------------------------------------------------
# Indicators and signals
# --------------------------------------------------------------------------
def adx_on_timeframe(df: pd.DataFrame, length: int, tf: str | None) -> pd.Series:
    """ADX on the chart timeframe, or on a higher timeframe without lookahead.

    Mirrors request.security(..., lookahead_off) on historical bars: a chart
    bar only sees a higher-timeframe ADX value once that HTF bar has closed.
    """
    if tf is None:
        return ind.adx(df, length)

    chart_step = pd.Series(df.index).diff().mode().iloc[0]
    htf = (
        df.resample(tf, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    htf_step = pd.Timedelta(tf)
    if htf_step < chart_step:
        raise ValueError(
            "ADX timeframe is lower than the chart timeframe. TradingView "
            "behaves inconsistently here, so it isn't supported."
        )
    htf_adx = ind.adx(htf, length).rename("adx")
    left = pd.DataFrame({"close_time": df.index + chart_step})
    right = pd.DataFrame({"close_time": htf.index + htf_step, "adx": htf_adx.to_numpy()})
    merged = pd.merge_asof(
        left.sort_values("close_time"), right.sort_values("close_time"),
        on="close_time", direction="backward",
    )
    return pd.Series(merged["adx"].values, index=df.index)


def compute_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    kc = ind.keltner(df, p.kc_length, p.kc_mult)
    out = out.join(kc)
    out["rsi"] = ind.rsi(df["close"], p.rsi_length)
    out["adx"] = adx_on_timeframe(df, p.adx_length, p.adx_tf)

    ranging = out["adx"] < p.adx_limit
    out["long_signal"] = (out["close"] < out["lower"]) & (out["rsi"] < p.rsi_os) & ranging
    out["short_signal"] = (out["close"] > out["upper"]) & (out["rsi"] > p.rsi_ob) & ranging

    # Exit levels exactly as written in the Pine script, from each bar's close
    risk = (out["close"] - out["basis"]).abs()
    out["long_tp"] = out["basis"]
    out["short_tp"] = out["basis"]
    out["long_sl"] = out["close"] - risk * p.risk_reward
    out["short_sl"] = out["close"] + risk * p.risk_reward
    return out


# --------------------------------------------------------------------------
# Fill logic
# --------------------------------------------------------------------------
def _exit_fill(direction: int, o: float, h: float, l: float, stop: float, limit: float):
    """Return (price, reason) if the stop or target fills on this bar, else None."""
    if np.isnan(stop) or np.isnan(limit):
        return None
    if direction == 1:  # long: sell limit above, sell stop below
        if o >= limit:
            return o, "tp_gap"
        if o <= stop:
            return o, "sl_gap"
        hit_tp, hit_sl = h >= limit, l <= stop
    else:  # short: buy limit below, buy stop above
        if o <= limit:
            return o, "tp_gap"
        if o >= stop:
            return o, "sl_gap"
        hit_tp, hit_sl = l <= limit, h >= stop

    if hit_tp and hit_sl:
        high_first = abs(h - o) < abs(l - o)
        # Long: target is up, stop is down. Short: the reverse.
        tp_first = high_first if direction == 1 else not high_first
        return (limit, "tp") if tp_first else (stop, "sl")
    if hit_tp:
        return limit, "tp"
    if hit_sl:
        return stop, "sl"
    return None


def backtest(df: pd.DataFrame, p: Params | None = None) -> Result:
    p = p or Params()
    bars = compute_signals(df, p)
    slip = p.slippage_ticks * p.tick_size

    opens = bars["open"].to_numpy()
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    closes = bars["close"].to_numpy()
    idx = bars.index

    trades: list[Trade] = []
    position = 0
    trade: Trade | None = None
    pending = 0  # direction of an entry order placed on the previous close
    stop = limit = np.nan  # levels placed on the previous close
    next_levels = {1: (np.nan, np.nan), -1: (np.nan, np.nan)}

    for i in range(len(bars)):
        # 1) Market entry placed last bar fills at this bar's open
        if pending and position == 0:
            position = pending
            fill = opens[i] + slip * position
            trade = Trade(position, idx[i], fill)
            stop, limit = next_levels[position]
        pending = 0

        # 2) Stop / target check on this bar
        if position != 0 and trade is not None:
            hit = _exit_fill(position, opens[i], highs[i], lows[i], stop, limit)
            if hit:
                price, reason = hit
                if reason.startswith("sl"):
                    price -= slip * position
                trade.exit_time, trade.exit_price, trade.exit_reason = idx[i], price, reason
                trade.pnl = (
                    (price - trade.entry_price) * position * p.point_value * p.qty
                    - 2 * p.commission_per_side * p.qty
                )
                trades.append(trade)
                position, trade = 0, None
            else:
                trade.bars_held += 1

        # 3) Script runs on the bar close: new entries, refresh exit levels
        row = bars.iloc[i]
        next_levels = {
            1: (row["long_sl"], row["long_tp"]),
            -1: (row["short_sl"], row["short_tp"]),
        }
        if position == 0:
            if row["long_signal"]:
                pending = 1
            elif row["short_signal"]:
                pending = -1
        else:
            new_stop, new_limit = next_levels[position]
            limit = new_limit  # target always follows the KC basis
            if not p.fixed_stop:
                stop = new_stop  # faithful: stop recalculated from latest close

    # Close anything still open at the last close so it shows up in the stats
    if trade is not None:
        price = closes[-1]
        trade.exit_time, trade.exit_price, trade.exit_reason = idx[-1], price, "end_of_data"
        trade.pnl = (
            (price - trade.entry_price) * position * p.point_value * p.qty
            - 2 * p.commission_per_side * p.qty
        )
        trades.append(trade)

    return Result(params=p, bars=bars, trades=trades)
