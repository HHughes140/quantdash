"""Seed the local DuckDB cache with prices + factors.

Usage: python scripts/seed_local.py [--tickers AAPL,MSFT,...] [--period 10y]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantdash.data.source import DuckDBSource
from quantdash.data.seed import seed
from quantdash.data.universe import DEFAULT_UNIVERSE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default=None, help="comma-separated; default = built-in universe")
    p.add_argument("--period", default="10y")
    p.add_argument("--no-factors", action="store_true")
    args = p.parse_args()

    tickers = ([t.strip().upper() for t in args.tickers.split(",")]
               if args.tickers else DEFAULT_UNIVERSE)
    src = DuckDBSource()
    stats = seed(src, tickers, period=args.period,
                 include_factors=not args.no_factors)
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
