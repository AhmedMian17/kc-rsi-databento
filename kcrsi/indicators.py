"""Indicators written to match TradingView Pine Script v6 behaviour.

Pine's ta.ema and ta.rma both seed with a simple moving average of the first
`length` values, then apply the recursive formula. pandas' ewm() does not do
this seeding, which makes early values drift from TradingView, so these are
implemented by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _seeded_recursive(src: pd.Series, length: int, alpha: float) -> pd.Series:
    """Shared engine for ta.ema / ta.rma: SMA seed, then alpha-weighted recursion."""
    values = src.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    start = None
    # Find the first window of `length` consecutive non-NaN values for the seed
    run = 0
    for i, v in enumerate(values):
        run = run + 1 if not np.isnan(v) else 0
        if run == length:
            start = i
            break
    if start is None:
        return pd.Series(out, index=src.index)
    out[start] = values[start - length + 1 : start + 1].mean()
    for i in range(start + 1, len(values)):
        v = values[i]
        out[i] = out[i - 1] if np.isnan(v) else alpha * v + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=src.index)


def ema(src: pd.Series, length: int) -> pd.Series:
    """ta.ema: alpha = 2 / (length + 1)."""
    return _seeded_recursive(src, length, 2.0 / (length + 1))


def rma(src: pd.Series, length: int) -> pd.Series:
    """ta.rma (Wilder's smoothing): alpha = 1 / length."""
    return _seeded_recursive(src, length, 1.0 / length)


def true_range(df: pd.DataFrame) -> pd.Series:
    """ta.tr: on the first bar Pine uses high - low (no previous close)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    """ta.atr = rma(true range)."""
    return rma(true_range(df), length)


def rsi(src: pd.Series, length: int) -> pd.Series:
    """ta.rsi: Wilder-smoothed average gains / losses."""
    change = src.diff()
    gain = rma(change.clip(lower=0), length)
    loss = rma((-change).clip(lower=0), length)
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    # Pine: loss == 0 -> 100, gain == 0 -> 0
    out = out.where(loss != 0, 100.0)
    out = out.where(gain != 0, 0.0)
    return out.where(gain.notna() & loss.notna())


def keltner(df: pd.DataFrame, length: int, mult: float) -> pd.DataFrame:
    """Keltner Channel exactly as in the Pine script: EMA basis +/- ATR * mult."""
    basis = ema(df["close"], length)
    rng = atr(df, length)
    return pd.DataFrame(
        {"basis": basis, "upper": basis + rng * mult, "lower": basis - rng * mult},
        index=df.index,
    )


def adx(df: pd.DataFrame, length: int) -> pd.Series:
    """Port of the custom f_adx() function in the Pine script (not ta.dmi)."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # ta.change() is NaN on the first bar, and Pine's ternary then yields 0
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = rma(true_range(df), length)
    plus = 100 * rma(plus_dm, length) / tr
    minus = 100 * rma(minus_dm, length) / tr
    dx = (plus - minus).abs() / (plus + minus) * 100
    return rma(dx, length)
