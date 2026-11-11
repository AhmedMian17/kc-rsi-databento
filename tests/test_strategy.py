import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kcrsi import indicators as ind  # noqa: E402
from kcrsi.data import resample  # noqa: E402
from kcrsi.strategy import Params, _exit_fill, backtest  # noqa: E402


def bars_from_close(close, spread=0.5, freq="30min"):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2026-01-05", periods=len(close), freq=freq, tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + spread,
         "low": np.minimum(open_, close) - spread, "close": close},
        index=idx,
    )


# ---------------------------------------------------------------- indicators
def test_ema_seeds_with_sma():
    s = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
    out = ind.ema(s, 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)          # SMA of 1,2,3
    assert out.iloc[3] == pytest.approx(0.5 * 4 + 0.5 * 2.0)  # alpha = 2/(3+1)


def test_rma_is_wilder():
    s = pd.Series([2, 4, 6, 8], dtype=float)
    out = ind.rma(s, 2)
    assert out.iloc[1] == pytest.approx(3.0)
    assert out.iloc[2] == pytest.approx(0.5 * 6 + 0.5 * 3.0)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert ind.rsi(up, 14).dropna().eq(100).all()
    rnd = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=300)) + 100)
    r = ind.rsi(rnd, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_adx_in_range():
    df = bars_from_close(np.cumsum(np.random.default_rng(1).normal(size=500)) + 2400)
    a = ind.adx(df, 14).dropna()
    assert len(a) > 400 and ((a >= 0) & (a <= 100)).all()


# ---------------------------------------------------------------- fills
def test_long_target_and_stop():
    assert _exit_fill(1, o=100, h=105, l=99, stop=95, limit=104) == (104, "tp")
    assert _exit_fill(1, o=100, h=101, l=94, stop=95, limit=104) == (95, "sl")
    assert _exit_fill(1, o=100, h=101, l=99, stop=95, limit=104) is None


def test_gap_fills_at_open():
    assert _exit_fill(1, o=93, h=94, l=92, stop=95, limit=104) == (93, "sl_gap")
    assert _exit_fill(-1, o=103, h=104, l=102, stop=110, limit=105) == (103, "tp_gap")


def test_both_hit_uses_tradingview_path():
    # High closer to open -> O-H-L-C, so a long hits its target first
    assert _exit_fill(1, o=100, h=104.5, l=94, stop=95, limit=104) == (104, "tp")
    # Low closer to open -> O-L-H-C, so a long hits its stop first
    assert _exit_fill(1, o=100, h=106, l=94.5, stop=95, limit=104) == (95, "sl")


# ---------------------------------------------------------------- engine
def test_entry_fills_next_open_and_exits_at_basis():
    # Flat market, a sharp drop (long signal), then a recovery
    # (small wiggle so ADX isn't 0/0, which is NaN in Pine too)
    close = [100 + 0.3 * (-1) ** k for k in range(60)] + [97, 94, 91] + [92, 95, 99, 101, 101, 101]
    df = bars_from_close(close, spread=0.2)
    res = backtest(df, Params(adx_limit=101))  # disable ADX filter for this test
    trades = res.trades_df()
    assert len(trades) >= 1
    first = trades.iloc[0]
    signal_bar = res.bars.index[res.bars["long_signal"]][0]
    entry_bar = res.bars.index[res.bars.index.get_loc(signal_bar) + 1]
    assert first["direction"] == 1
    assert first["entry_time"] == entry_bar
    assert first["entry_price"] == df.loc[entry_bar, "open"]
    assert first["exit_reason"] in {"tp", "tp_gap"}
    assert first["pnl"] > 0


def test_costs_reduce_pnl():
    df = bars_from_close(np.cumsum(np.random.default_rng(3).normal(0, 1.5, 3000)) + 2400)
    gross = backtest(df, Params())
    net = backtest(df, Params(commission_per_side=0.62, slippage_ticks=1))
    if gross.trades:
        assert net.trades_df()["pnl"].sum() < gross.trades_df()["pnl"].sum()


def test_higher_tf_adx_has_no_lookahead():
    df = bars_from_close(np.cumsum(np.random.default_rng(4).normal(size=400)) + 2400)
    full = backtest(df, Params(adx_tf="2h")).bars["adx"]
    cut = backtest(df.iloc[:250], Params(adx_tf="2h")).bars["adx"]
    # Values up to bar 250 must not change when future data is added
    pd.testing.assert_series_equal(full.iloc[:250], cut, check_names=False)


def test_resample_ohlc():
    idx = pd.date_range("2026-01-05", periods=60, freq="1min", tz="UTC")
    df = pd.DataFrame({"open": range(60), "high": range(1, 61), "low": range(60),
                       "close": range(60), "volume": 1}, index=idx, dtype=float)
    r = resample(df, "30min")
    assert len(r) == 2
    assert r.iloc[0]["open"] == 0 and r.iloc[0]["close"] == 29 and r.iloc[0]["high"] == 30
    assert r.iloc[0]["volume"] == 30
