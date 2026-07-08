"""Seed a DataSource (DuckDB or Snowflake) with prices and factor returns.

Prices come from yfinance; factors from the Ken French data library
(FF5 daily + daily momentum), stored as decimal daily returns.
"""

from __future__ import annotations

import io
import zipfile
from typing import Optional, Sequence

import pandas as pd
import requests

from .source import DataSource
from .universe import BENCHMARKS, DEFAULT_UNIVERSE

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF5_DAILY = f"{FRENCH_BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_DAILY = f"{FRENCH_BASE}/F-F_Momentum_Factor_daily_CSV.zip"


def fetch_prices(
    tickers: Sequence[str], period: str = "10y", interval: str = "1d"
) -> pd.DataFrame:
    """Download OHLCV for tickers, return long-format PRICES rows."""
    import yfinance as yf

    raw = yf.download(
        list(tickers),
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            sub = raw[t].dropna(how="all")
            if sub.empty:
                continue
            frames.append(_to_long(sub, t))
    else:  # single ticker
        frames.append(_to_long(raw.dropna(how="all"), tickers[0]))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["adj_close"])


def _to_long(sub: pd.DataFrame, ticker: str) -> pd.DataFrame:
    cols = {c.lower().replace(" ", "_"): c for c in sub.columns}
    adj = cols.get("adj_close", cols.get("close"))
    return pd.DataFrame(
        {
            "date": pd.to_datetime(sub.index).date,
            "ticker": ticker,
            "open": sub[cols["open"]].values,
            "high": sub[cols["high"]].values,
            "low": sub[cols["low"]].values,
            "close": sub[cols["close"]].values,
            "adj_close": sub[adj].values,
            "volume": sub[cols["volume"]].values,
        }
    )


def _read_french_zip(url: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV zip into a date-indexed DataFrame (decimals)."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    text = zf.read(zf.namelist()[0]).decode("latin-1")

    lines = text.splitlines()
    # Header row = first line with an empty first field and at least one named column
    start = None
    for i, line in enumerate(lines):
        parts = line.split(",")
        if len(parts) > 1 and parts[0].strip() == "" and any(p.strip() for p in parts[1:]):
            start = i
            break
    if start is None:
        raise ValueError(f"Could not locate header in {url}")

    rows, header = [], [h.strip() for h in lines[start].split(",")]
    for line in lines[start + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(header) or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:  # ran past the daily block (annual section / copyright)
                break
            continue
        rows.append(parts)

    # Name columns positionally; blank header cells (e.g. trailing commas) get
    # placeholder names and are dropped so we never end up with duplicates.
    names = ["date"] + [h if h else f"_blank{i}" for i, h in enumerate(header[1:], 1)]
    df = pd.DataFrame(rows, columns=names)
    df = df.drop(columns=[c for c in names[1:] if c.startswith("_blank")])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df.set_index("date")


def fetch_ff_factors() -> pd.DataFrame:
    """FF5 + momentum daily factors, long format (date, factor, value)."""
    ff5 = _read_french_zip(FF5_DAILY)
    mom = _read_french_zip(MOM_DAILY)
    ff5.columns = [c.upper().replace("-", "_").replace(" ", "") for c in ff5.columns]
    mom.columns = ["MOM" for _ in mom.columns]
    merged = ff5.join(mom, how="left")
    merged = merged.rename(columns={"MKT_RF": "MKT_RF"})
    long = merged.reset_index().melt(id_vars="date", var_name="factor", value_name="value")
    long["date"] = long["date"].dt.date
    return long.dropna(subset=["value"])


def seed(
    source: DataSource,
    tickers: Optional[Sequence[str]] = None,
    period: str = "10y",
    include_factors: bool = True,
    log=print,
) -> dict:
    """Seed prices (+benchmark ETFs) and factors into the given source."""
    tickers = list(dict.fromkeys(list(tickers or DEFAULT_UNIVERSE) + BENCHMARKS))
    log(f"[seed] downloading {len(tickers)} tickers, period={period} ...")
    prices = fetch_prices(tickers, period=period)
    n_prices = source.write_prices(prices)
    log(f"[seed] wrote {n_prices:,} price rows for "
        f"{prices['ticker'].nunique() if not prices.empty else 0} tickers")

    n_factors = 0
    if include_factors:
        log("[seed] downloading Fama-French daily factors (FF5 + MOM) ...")
        factors = fetch_ff_factors()
        n_factors = source.write_factors(factors)
        log(f"[seed] wrote {n_factors:,} factor rows "
            f"({factors['factor'].nunique()} factors)")
    return {"price_rows": n_prices, "factor_rows": n_factors}
