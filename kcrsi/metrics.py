"""Summary statistics in the same terms as the TradingView Strategy Tester."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import Result


def summarise(res: Result) -> dict:
    t = res.trades_df()
    if t.empty:
        return {"trades": 0}

    pnl = t["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    equity = res.params.initial_capital + pnl.cumsum()
    peak = equity.cummax().clip(lower=res.params.initial_capital)
    drawdown = peak - equity

    def side(direction: int) -> dict:
        s = t[t["direction"] == direction]["pnl"]
        return {
            "trades": int(len(s)),
            "win_rate_pct": round(100 * (s > 0).mean(), 1) if len(s) else None,
            "net_profit": round(s.sum(), 2),
        }

    gross_loss = -losses.sum()
    return {
        "trades": int(len(t)),
        "win_rate_pct": round(100 * len(wins) / len(t), 1),
        "net_profit": round(pnl.sum(), 2),
        "profit_factor": round(wins.sum() / gross_loss, 2) if gross_loss > 0 else np.inf,
        "avg_trade": round(pnl.mean(), 2),
        "avg_win": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(losses.mean(), 2) if len(losses) else 0.0,
        "max_drawdown": round(drawdown.max(), 2),
        "avg_bars_held": round(t["bars_held"].mean(), 1),
        "exit_reasons": t["exit_reason"].value_counts().to_dict(),
        "long": side(1),
        "short": side(-1),
    }


def equity_curve(res: Result) -> pd.Series:
    t = res.trades_df()
    if t.empty:
        return pd.Series(dtype=float)
    return pd.Series(
        res.params.initial_capital + t["pnl"].cumsum().values,
        index=pd.to_datetime(t["exit_time"]),
        name="equity",
    )
