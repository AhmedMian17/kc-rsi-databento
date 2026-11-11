"""Run the KC + RSI Mean Reversion backtest on Databento MGC data.

Usage:
    python scripts/run_backtest.py                      # data/mgc_1m.csv, 30m bars
    python scripts/run_backtest.py --timeframe 15min --adx-tf 1h
    python scripts/run_backtest.py --synthetic          # demo run, no API key needed
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from kcrsi.data import flag_rolls, load_bars, resample  # noqa: E402
from kcrsi.metrics import equity_curve, summarise  # noqa: E402
from kcrsi.strategy import Params, backtest  # noqa: E402


def synthetic_bars(n_minutes: int = 60 * 24 * 90, seed: int = 7) -> pd.DataFrame:
    """Random-walk 1-minute bars for testing the pipeline without real data."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-04 23:00", periods=n_minutes, freq="1min", tz="UTC")
    close = 2400 + np.cumsum(rng.normal(0, 0.6, n_minutes))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.4, n_minutes))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.integers(1, 50, n_minutes),
        },
        index=idx,
    )


def plot(res, eq_by_name: dict[str, pd.Series], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, eq in eq_by_name.items():
        if not eq.empty:
            ax.step(eq.index, eq.values, where="post", label=name, linewidth=1.6)
    ax.axhline(res.params.initial_capital, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("KC + RSI Mean Reversion on MGC — equity (1 contract)")
    ax.set_ylabel("Account equity (USD)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    # Price chart of the last ~300 bars with the channel and trades
    bars = res.bars.iloc[-300:]
    trades = res.trades_df()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(bars.index, bars["close"], color="black", linewidth=0.9, label="Close")
    ax.plot(bars.index, bars["basis"], color="tab:orange", linewidth=1, label="KC basis (EMA)")
    ax.plot(bars.index, bars["upper"], color="tab:blue", linewidth=0.8, label="KC bands")
    ax.plot(bars.index, bars["lower"], color="tab:blue", linewidth=0.8)
    if not trades.empty:
        recent = trades[pd.to_datetime(trades["entry_time"]) >= bars.index[0]]
        for _, t in recent.iterrows():
            colour = "tab:green" if t["direction"] == 1 else "tab:red"
            marker = "^" if t["direction"] == 1 else "v"
            ax.scatter(t["entry_time"], t["entry_price"], color=colour, marker=marker, s=50, zorder=3)
            ax.scatter(t["exit_time"], t["exit_price"], color="grey", marker="x", s=35, zorder=3)
    ax.set_title("Last 300 bars — entries (triangles) and exits (crosses)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "recent_trades.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/mgc_1m.csv")
    ap.add_argument("--synthetic", action="store_true", help="Use random-walk data instead")
    ap.add_argument("--timeframe", default="30min", help="Chart timeframe (default 30min)")
    ap.add_argument("--adx-tf", default=None, help="ADX timeframe, default = chart timeframe")
    ap.add_argument("--rr", type=float, default=1.0, help="Risk Reward input (default 1.0)")
    ap.add_argument("--commission", type=float, default=0.62, help="USD per contract per side for the 'with costs' run")
    ap.add_argument("--slippage", type=int, default=1, help="Ticks of slippage for the 'with costs' run")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    raw = synthetic_bars() if args.synthetic else load_bars(Path(args.data))
    bars = resample(raw, args.timeframe)
    rolls = flag_rolls(bars)
    print(f"{len(bars):,} {args.timeframe} bars, {bars.index[0]} -> {bars.index[-1]}")
    if rolls.any():
        print(f"Contract rolls at: {', '.join(str(t) for t in bars.index[rolls])}")

    base = Params(adx_tf=args.adx_tf, risk_reward=args.rr)
    costs = dict(commission_per_side=args.commission, slippage_ticks=args.slippage)
    runs = {
        "Pine-faithful (no costs)": base,
        "Pine-faithful (with costs)": replace(base, **costs),
        "Fixed stop (with costs)": replace(base, fixed_stop=True, **costs),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, curves, first = {}, {}, None
    for name, params in runs.items():
        res = backtest(bars, params)
        first = first or res
        summary[name] = summarise(res)
        curves[name] = equity_curve(res)
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        res.trades_df().to_csv(out_dir / f"trades_{slug}.csv", index=False)

    print(json.dumps(summary, indent=2, default=str))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot(first, curves, out_dir)
    print(f"Charts, trades and summary written to {out_dir}/")


if __name__ == "__main__":
    main()
