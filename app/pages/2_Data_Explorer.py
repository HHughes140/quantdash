"""Data Explorer — coverage, seeding, and Snowflake connection status."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from quantdash.data import get_source
from quantdash.data.source import DuckDBSource, SnowflakeConfig, SnowflakeSource
from quantdash.data.seed import seed
from quantdash.data.universe import DEFAULT_UNIVERSE
from quantdash.ui import inject_css

st.set_page_config(page_title="Data Explorer", page_icon="🗄️", layout="wide")
inject_css()
st.title("🗄️ Data Explorer")


@st.cache_resource
def _source():
    return get_source()


src = _source()

# ---------------- Connection status ----------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Active source")
    st.metric("Backend", src.name)
    if src.name == "duckdb":
        st.caption(f"Local DuckDB at `{src.db_path}`")
with c2:
    st.subheader("Snowflake")
    cfg = SnowflakeConfig.from_env()
    if cfg is None:
        st.warning(
            "Not configured. Set env vars (or `.streamlit/secrets.toml`):\n\n"
            "`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD` "
            "(or `SNOWFLAKE_PRIVATE_KEY_PATH`), `SNOWFLAKE_WAREHOUSE`, "
            "`SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`"
        )
    else:
        st.success(f"Configured: {cfg.account} / {cfg.database}.{cfg.schema}")
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
    st.dataframe(cov, use_container_width=True, height=300)

facs = src.available_factors()
st.subheader("Factors")
st.write(", ".join(f"`{f}`" for f in facs) if facs else "None loaded.")

st.divider()

# ---------------- Seeding ----------------
st.subheader("Seed / refresh data")
target = st.radio("Target", ["Active source", "Snowflake (explicit)"], horizontal=True)
uni = st.text_area(
    "Tickers (comma-separated; SPY is always added as benchmark)",
    value=", ".join(DEFAULT_UNIVERSE), height=120,
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
        dest = src
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
