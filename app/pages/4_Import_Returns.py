"""Import Returns — analyze external daily return streams with the lab's
metrics and factor overlays.

Accepts a generic CSV (date + return column) or a book/factor L/S export in
the ff32 format (DDATE, BOOK, factor/raw_factor_column, long_short_return):
if BOOK/factor columns are present you can filter to one strategy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from quantdash.data import get_source, get_theory_store
from quantdash.engine.factors import factor_regression, rolling_factor_betas
from quantdash.engine.metrics import compute_metrics, drawdown_series
from quantdash.ui import (ACCENT, GRAY, GREEN, RED, factor_color, inject_css,
                          page_header, style_fig)

st.set_page_config(page_title="Import Returns — Insurance Alpha Lab",
                   page_icon="📥", layout="wide")
inject_css()
page_header("Import Returns",
            "Book exports, live P&L, external strategies — same lens as the lab")


@st.cache_resource
def _source():
    return get_source()


@st.cache_data(ttl=3600, show_spinner=False)
def _factors() -> pd.DataFrame:
    return _source().get_factors()


up = st.file_uploader(
    "Daily returns CSV — generic (date, return) or ff32 book export "
    "(DDATE, BOOK, factor, long_short_return)", type=["csv"])
if up is None:
    st.info("Upload a CSV to analyze it.")
    st.stop()

df = pd.read_csv(up)
if df.empty:
    st.error("Empty file.")
    st.stop()

date_col = next((c for c in df.columns
                 if str(c).lower() in ("ddate", "date", "dt", "day")),
                df.columns[0])
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])

# Optional strategy filters (ff32-style exports)
fc1, fc2, fc3, fc4 = st.columns(4)
book_col = next((c for c in df.columns if str(c).upper() == "BOOK"), None)
if book_col:
    books = ["(all)"] + sorted(df[book_col].astype(str).unique())
    pick_book = fc1.selectbox("Book", books)
    if pick_book != "(all)":
        df = df[df[book_col].astype(str).eq(pick_book)]
factor_col = next((c for c in df.columns
                   if str(c).lower() in ("factor", "raw_factor_column")), None)
if factor_col:
    facs = ["(all)"] + sorted(df[factor_col].astype(str).unique())
    pick_fac = fc2.selectbox("Factor / strategy", facs)
    if pick_fac != "(all)":
        df = df[df[factor_col].astype(str).eq(pick_fac)]

num_cols = [c for c in df.columns
            if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
if not num_cols:
    st.error("No numeric return column found.")
    st.stop()
default_ret = ("long_short_return" if "long_short_return" in num_cols
               else num_cols[0])
ret_col = fc3.selectbox("Return column", num_cols,
                        index=num_cols.index(default_ret))
in_pct = fc4.checkbox("Values are in percent", value=False)

series = pd.to_numeric(df.set_index(date_col)[ret_col], errors="coerce") \
    .groupby(level=0).mean().sort_index().dropna()
if in_pct:
    series = series / 100.0
if len(series) < 40:
    st.error(f"Only {len(series)} daily observations — need at least 40.")
    st.stop()

label_default = Path(up.name).stem
label = st.text_input("Label", value=label_default)

# ---------------- Metrics ----------------
m = compute_metrics(series)
if "error" in m:
    st.error(m["error"])
    st.stop()

row = st.columns(6)
row[0].metric("Sharpe", f"{m['sharpe']:.2f}")
row[1].metric("Lo-corrected", f"{m['sharpe_lo_corrected']:.2f}",
              f"p={m['sharpe_p_value']:.3f}", delta_color="off")
row[2].metric("Ann. return", f"{m['ann_return']:.1%}")
row[3].metric("Ann. vol", f"{m['ann_vol']:.1%}")
row[4].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
row[5].metric("Hit rate", f"{m['hit_rate']:.1%}")
st.caption(f"{m['n_days']:,} trading days · {m['start']} → {m['end']}")

eq = (1 + series).cumprod()
dd = drawdown_series(series)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.74, 0.26], vertical_spacing=0.03)
fig.add_trace(go.Scatter(x=eq.index, y=eq, name=label,
                         line=dict(width=2.2, color=ACCENT),
                         hovertemplate="%{y:.2f}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", showlegend=False,
                         line=dict(color=RED, width=1),
                         fillcolor="rgba(240,84,79,0.22)",
                         hovertemplate="%{y:.1%}<extra>drawdown</extra>"),
              row=2, col=1)
style_fig(fig, height=480, title=f"{label} — growth of $1")
fig.update_yaxes(tickformat=".0%", row=2, col=1)
st.plotly_chart(fig, width="stretch")

# ---------------- Factor overlay ----------------
factors = _factors()
if factors.empty:
    st.warning("No factor data available from the active source.")
else:
    st.subheader("Factor overlay")
    avail = [c for c in factors.columns if str(c).upper() != "RF"]
    chosen = st.multiselect("Factors", avail, default=avail, key="imp_facs")
    fac = factors[chosen + (["RF"] if "RF" in factors.columns else [])]
    reg = factor_regression(series, fac)
    if "error" in reg:
        st.warning(reg["error"])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Factor α (ann)", f"{reg['alpha_ann']:.2%}",
                  f"t = {reg['alpha_tstat']:.2f}", delta_color="off")
        c2.metric("α p-value", f"{reg['alpha_pvalue']:.3f}",
                  "significant" if reg["alpha_pvalue"] < 0.05 else "not significant",
                  delta_color="off")
        c3.metric("R²", f"{reg['r_squared']:.2f}")
        beta_df = pd.DataFrame(reg["betas"]).T
        beta_df.columns = ["beta", "t-stat", "p-value"]
        st.dataframe(
            beta_df.style.format({"beta": "{:.3f}", "t-stat": "{:.2f}",
                                  "p-value": "{:.3f}"})
            .map(lambda v: "background-color: rgba(255,165,0,.25)"
                 if isinstance(v, float) and abs(v) > 2 else "",
                 subset=["t-stat"]),
            width="stretch")

        roll = rolling_factor_betas(series, fac, window=126)
        if not roll.empty:
            fig_b = go.Figure()
            beta_cols = [c for c in roll.columns
                         if c not in ("alpha_ann", "r_squared")]
            for i, col in enumerate(beta_cols):
                fig_b.add_trace(go.Scatter(
                    x=roll.index, y=roll[col], name=col,
                    line=dict(width=2, color=factor_color(col, i)),
                    hovertemplate="%{y:.2f}<extra>" + col + "</extra>"))
            style_fig(fig_b, height=360, title="Rolling 6-month factor betas")
            st.plotly_chart(fig_b, width="stretch")

# ---------------- Actions ----------------
st.divider()
a1, a2 = st.columns(2)
if a1.button("➕ Add to Compare (Backtest Lab tab)", width="stretch"):
    snap_metrics = {k: v for k, v in m.items() if not isinstance(v, (tuple, str))}
    st.session_state.setdefault("compare", {})[label] = {
        "returns": series, "metrics": snap_metrics,
        "expression": f"[imported] {up.name}", "config": {"source": "import"},
    }
    a1.success("Added — open the Compare tab on the Backtest Lab page.")
if a2.button("💾 Save to Theory Journal", width="stretch"):
    keep = ["sharpe", "sharpe_lo_corrected", "sharpe_p_value", "ann_return",
            "ann_vol", "max_drawdown", "hit_rate", "start", "end", "n_days"]
    get_theory_store(_source()).save_theory(
        name=label, hypothesis=f"Imported returns from {up.name}",
        expression=f"[imported] {up.name}", config={"source": "import",
                                                    "return_col": ret_col},
        metrics={k: m[k] for k in keep if k in m},
        verdict=("supported" if m["sharpe_p_value"] < 0.05 and m["sharpe"] > 0
                 else "not supported"))
    a2.success("Saved to the Theory Journal.")
