"""Data Explorer — coverage, seeding, and Snowflake connection status."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from quantdash.data import get_source, get_theory_store
from quantdash.data.source import DuckDBSource, SnowflakeConfig, SnowflakeSource
from quantdash.data.seed import seed
from quantdash.data.universe import BENCHMARKS, DEFAULT_UNIVERSE, INSURANCE_TICKERS
from quantdash.engine.snowflake_utilities import HAVE_FIRM_UTILS
from quantdash.ui import inject_css, page_header

st.set_page_config(page_title="Data Explorer — Insurance Alpha Lab",
                   page_icon="🗄️", layout="wide")
inject_css()
page_header("Data Explorer", "Sources · coverage · seeding")


@st.cache_resource
def _source():
    return get_source()


src = _source()
read_only = getattr(src, "read_only", False)

# ---------------- Connection status ----------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Active source")
    st.metric("Backend", src.name)
    if src.name == "duckdb":
        st.caption(f"Local DuckDB at `{src.db_path}`")
    elif read_only:
        st.caption("Read-only Axioma feed: prices = cumulated `_1_DAY_RETURN` "
                   "(WW4/SH), volume = `_20_DAY_ADV`, factors = "
                   "`AXIOMA.FUNDAMENTAL.FACTOR_RETURN`. Theories persist to the "
                   f"local DuckDB store (`{get_theory_store(src).db_path}`).")
with c2:
    st.subheader("Snowflake")
    if HAVE_FIRM_UTILS:
        st.success("Firm `snowflake_utilities` detected — Axioma tables available "
                   "(warehouse WHSE_TEAM_WILHELM_001).")
    cfg = SnowflakeConfig.from_env()
    if cfg is None and not HAVE_FIRM_UTILS:
        st.warning(
            "Not configured. Either install the firm `snowflake_utilities` "
            "package (Axioma path) or set env vars: `SNOWFLAKE_ACCOUNT`, "
            "`SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD` (or "
            "`SNOWFLAKE_PRIVATE_KEY_PATH`), `SNOWFLAKE_WAREHOUSE`, "
            "`SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`.\n\n"
            "Force a specific backend with `QUANTDASH_SOURCE=axioma|snowflake|duckdb`."
        )
    elif cfg is not None:
        st.success(f"Env credentials: {cfg.account} / {cfg.database}.{cfg.schema}")
        if st.button("Test connection"):
            try:
                sf = SnowflakeSource(cfg)
                n = len(sf.available_tickers())
                st.success(f"Connected. PRICES has {n} tickers.")
            except Exception as e:
                st.error(f"Connection failed: {e}")

st.divider()

# ---------------- Coverage ----------------
st.subheader("Price coverage")
cov = src.coverage()
if cov.empty:
    st.info("No price data loaded yet — seed below.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Tickers", len(cov))
    c2.metric("Date range", f"{cov['start'].min()} → {cov['end'].max()}")
    c3.metric("Rows", f"{int(cov['rows'].sum()):,}")
    st.dataframe(cov, width="stretch", height=300)

facs = src.available_factors()
st.subheader("Factors")
st.write(", ".join(f"`{f}`" for f in facs) if facs else "None loaded.")

with st.expander("Upload custom factor returns"):
    st.caption(
        "Wide CSV: first column = date, one column per factor, daily returns. "
        "Stored in the local DuckDB and **merged into Factor Overlays "
        "automatically** alongside the source factors — this is the drop-in "
        "for custom factor loader exports."
    )
    fup = st.file_uploader("Factor CSV", type=["csv"], key="factor_up")
    fpct = st.checkbox("Values are in percent", key="factor_pct")
    if fup is not None and st.button("Load factors", key="factor_load"):
        fdf = pd.read_csv(fup)
        datec = fdf.columns[0]
        fdf[datec] = pd.to_datetime(fdf[datec], errors="coerce")
        fdf = fdf.dropna(subset=[datec])
        long = fdf.melt(id_vars=datec, var_name="factor", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        if fpct:
            long["value"] = long["value"] / 100.0
        long = long.dropna(subset=["value"])
        long["date"] = long[datec].dt.date
        if long.empty:
            st.error("No parsable rows — check the format.")
        else:
            n = DuckDBSource().write_factors(long[["date", "factor", "value"]])
            st.success(f"Loaded {n:,} rows across "
                       f"{long['factor'].nunique()} factors: "
                       + ", ".join(f"`{f}`" for f in sorted(long["factor"].unique())))
            st.cache_data.clear()

st.divider()

# ---------------- Macro data ----------------
st.subheader("Macro data")
_macro_store = DuckDBSource()
macro_names = _macro_store.available_macro()
if macro_names:
    mp = _macro_store.get_macro_panel()
    mc1, mc2 = st.columns([1, 2])
    mc1.metric("Series", len(macro_names))
    mc2.caption(", ".join(f"`{s}`" for s in macro_names)
                + f" · {mp.index.min().date()} → {mp.index.max().date()}")
else:
    st.caption("No macro series loaded yet. Upload a workbook below to use "
               "`macro(\"NAME\")` in signal expressions.")

with st.expander("Upload macro workbook (Excel or CSV)", expanded=not macro_names):
    st.caption(
        "One date column + one column per series (levels). Excel: pick the "
        "sheet. Series become available in the DSL as `macro(\"NAME\")`, "
        "`macro_z(\"NAME\", n)`, `macro_chg(\"NAME\", n)` — values are "
        "forward-filled onto trading days, so monthly/weekly data is fine."
    )
    mup = st.file_uploader("Workbook", type=["xlsx", "xls", "csv"], key="macro_up")
    if mup is not None:
        if mup.name.lower().endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(mup)
            sheet = st.selectbox("Sheet", xl.sheet_names, key="macro_sheet")
            mdf = xl.parse(sheet)
        else:
            mdf = pd.read_csv(mup)
        if mdf.empty:
            st.error("Empty file/sheet.")
        else:
            st.dataframe(mdf.head(5), width="stretch")
            cols = list(mdf.columns)
            date_guess = next(
                (c for c in cols if str(c).lower() in
                 ("date", "ddate", "dt", "day", "month", "period")), cols[0])
            datec = st.selectbox("Date column", cols,
                                 index=cols.index(date_guess), key="macro_date")
            numeric = [c for c in cols if c != datec and
                       pd.to_numeric(mdf[c], errors="coerce").notna().mean() > 0.5]
            series_cols = st.multiselect("Series columns", numeric,
                                         default=numeric, key="macro_cols")
            if st.button("Load macro series", key="macro_load") and series_cols:
                mdf[datec] = pd.to_datetime(mdf[datec], errors="coerce")
                mdf = mdf.dropna(subset=[datec])
                long = mdf.melt(id_vars=datec, value_vars=series_cols,
                                var_name="series", value_name="value")
                long["value"] = pd.to_numeric(long["value"], errors="coerce")
                long = long.dropna(subset=["value"])
                # normalize names to DSL-friendly identifiers
                long["series"] = (long["series"].astype(str).str.strip()
                                  .str.upper().str.replace(r"[^A-Z0-9]+", "_",
                                                           regex=True)
                                  .str.strip("_"))
                long["date"] = long[datec].dt.date
                if long.empty:
                    st.error("No parsable rows.")
                else:
                    n = _macro_store.write_macro(long[["date", "series", "value"]])
                    st.success(
                        f"Loaded {n:,} rows across "
                        f"{long['series'].nunique()} series: "
                        + ", ".join(f"`{s}`"
                                    for s in sorted(long["series"].unique())))
                    st.cache_data.clear()

st.divider()

# ---------------- Seeding ----------------
st.subheader("Seed / refresh data")
if read_only:
    st.info("The active Axioma source is **read-only** — nothing to seed. "
            "Seeding below targets the **local DuckDB cache** so the lab also "
            "works offline (`QUANTDASH_SOURCE=duckdb`).")
target = st.radio(
    "Target",
    ["Local DuckDB" if read_only else "Active source", "Snowflake (explicit)"],
    horizontal=True)
uni = st.text_area(
    "Tickers (comma-separated; benchmarks SPY/KIE/IAK are always added)",
    value=", ".join(INSURANCE_TICKERS + DEFAULT_UNIVERSE), height=120,
)
period = st.selectbox("History", ["2y", "5y", "10y", "max"], index=2)
with_factors = st.checkbox("Also load Fama-French factors (FF5 + MOM)", value=True)

if st.button("🌱 Seed now", type="primary"):
    tickers = [t.strip().upper() for t in uni.replace("\n", ",").split(",") if t.strip()]
    if target == "Snowflake (explicit)":
        cfg = SnowflakeConfig.from_env()
        if cfg is None:
            st.error("Snowflake env vars not set.")
            st.stop()
        dest = SnowflakeSource(cfg)
    else:
        dest = DuckDBSource() if read_only else src
    prog = st.status(f"Seeding {len(tickers)} tickers into {dest.name}...",
                     expanded=True)
    try:
        stats = seed(dest, tickers, period=period, include_factors=with_factors,
                     log=prog.write)
        prog.update(label="Done", state="complete")
        st.success(f"Wrote {stats['price_rows']:,} price rows and "
                   f"{stats['factor_rows']:,} factor rows to {dest.name}.")
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception as e:
        prog.update(state="error")
        st.error(f"Seed failed: {e}")
