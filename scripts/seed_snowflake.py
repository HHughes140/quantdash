"""Seed Snowflake with prices + factors (system of record).

Requires env vars: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
(or SNOWFLAKE_PRIVATE_KEY_PATH), SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE,
SNOWFLAKE_SCHEMA. Creates PRICES / FACTORS / THEORIES tables if missing.

Usage: python scripts/seed_snowflake.py [--tickers ...] [--period 10y]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantdash.data.source import SnowflakeConfig, SnowflakeSource
from quantdash.data.seed import seed
from quantdash.data.universe import DEFAULT_UNIVERSE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default=None)
    p.add_argument("--period", default="10y")
    p.add_argument("--no-factors", action="store_true")
    args = p.parse_args()

    cfg = SnowflakeConfig.from_env()
    if cfg is None:
        sys.exit("SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER not set — see module docstring.")

    tickers = ([t.strip().upper() for t in args.tickers.split(",")]
               if args.tickers else DEFAULT_UNIVERSE)
    src = SnowflakeSource(cfg)
    stats = seed(src, tickers, period=args.period,
                 include_factors=not args.no_factors)
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
