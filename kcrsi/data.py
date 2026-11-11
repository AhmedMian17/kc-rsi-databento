"""Download MGC futures bars from Databento and resample them.

Databento publishes 1-second, 1-minute, 1-hour and 1-day OHLCV schemas, so a
30-minute chart is built by resampling 1-minute bars.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATASET = "GLBX.MDP3"  # CME Globex
SCHEMA = "ohlcv-1m"


def get_client():
    import databento as db  # imported here so tests run without the package

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError(
            "Set DATABENTO_API_KEY first (Databento portal > API Keys), e.g.\n"
            "  export DATABENTO_API_KEY=db-xxxxxxxx"
        )
    return db.Historical(key)


def estimate_cost(symbol: str, start: str, end: str, stype_in: str = "continuous") -> float:
    """Ask Databento what the request will cost (USD) before downloading."""
    client = get_client()
    return client.metadata.get_cost(
        dataset=DATASET, symbols=[symbol], schema=SCHEMA,
        stype_in=stype_in, start=start, end=end,
    )


def download_bars(symbol: str, start: str, end: str, out_path: Path,
                  stype_in: str = "continuous") -> pd.DataFrame:
    """Fetch 1-minute bars and cache them to CSV.

    symbol: a continuous contract such as "MGC.v.0" (rolls to the most-traded
    contract by volume) or "MGC.c.0" (rolls by calendar/expiry).
    """
    client = get_client()
    store = client.timeseries.get_range(
        dataset=DATASET, symbols=[symbol], schema=SCHEMA,
        stype_in=stype_in, start=start, end=end,
    )
    df = store.to_df()  # prices come back as floats, indexed by ts_event (UTC)
    df = df[["open", "high", "low", "close", "volume", "instrument_id"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return df


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "ts"
    return df.sort_index()


def resample(df: pd.DataFrame, rule: str = "30min") -> pd.DataFrame:
    """Build higher-timeframe bars. Empty periods (market closed) are dropped.

    CME's trading day starts at 17:00 Chicago time, which falls on a UTC hour,
    so 30-minute and 1-hour bins line up with TradingView. For 4h and daily
    bars TradingView anchors to the session open, so expect small differences.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "instrument_id" in df.columns:
        agg["instrument_id"] = "last"
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])


def flag_rolls(df: pd.DataFrame) -> pd.Series:
    """True on bars where the continuous contract switched to a new expiry.

    Continuous symbols are not back-adjusted, so the price can jump at a roll.
    Worth knowing when you see an odd trade around the roll dates.
    """
    if "instrument_id" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["instrument_id"].ne(df["instrument_id"].shift()) & df.index.to_series().ne(df.index[0])
