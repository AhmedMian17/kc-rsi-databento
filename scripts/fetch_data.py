"""Download MGC 1-minute bars from Databento (with a cost check first).

Usage:
    export DATABENTO_API_KEY=db-xxxxxxxx
    python scripts/fetch_data.py --start 2026-05-01 --end 2026-11-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kcrsi.data import download_bars, estimate_cost  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="MGC.v.0", help="Continuous symbol (default: MGC.v.0)")
    ap.add_argument("--start", required=True, help="Start date, e.g. 2026-05-01")
    ap.add_argument("--end", required=True, help="End date (exclusive), e.g. 2026-11-01")
    ap.add_argument("--out", default="data/mgc_1m.csv")
    ap.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt")
    args = ap.parse_args()

    cost = estimate_cost(args.symbol, args.start, args.end)
    print(f"Estimated cost for {args.symbol} {args.start} -> {args.end}: ${cost:.2f} USD")
    if not args.yes and input("Download? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return

    df = download_bars(args.symbol, args.start, args.end, Path(args.out))
    print(f"Saved {len(df):,} one-minute bars to {args.out}")
    print(f"Range: {df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()
