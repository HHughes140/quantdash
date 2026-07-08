"""Risk — factor exposure and stress testing for the current book.

Uses the latest backtest weights (run one on the Backtest Lab page first) or an
uploaded weights CSV. On the desk with `snowflake_utilities` installed, the
live Axioma z-scored exposures are available; elsewhere exposures come from
trailing regression betas — same shape, clearly labeled.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantdash.data import get_source
from quantdash.engine.factors import factor_exposure_heatmap_data
from quantdash.engine.snowflake_utilities import HAVE_FIRM_UTILS
from quantdash.ui import (ACCENT, GRAY, GREEN, RED, diverging_colors,
                          factor_color, inject_css, page_header, style_fig)

st.set_page_config(page_title="Risk — Insurance Alpha Lab", page_icon="🛡️",
                   layout="wide")
inject_css()
page_header("Risk", "Factor exposures and stress tests for the current book",
            badge="live Axioma" if HAVE_FIRM_UTILS else "regression betas")


@st.cache_resource
def _source():
    return get_source()


@st.cache_data(ttl=3600, show_spinner=False)
def _factors():
    return _source().get_factors()


# ---------------- Get a book ----------------
weights = None
result = st.session_state.get("result")
src_label = None
if result is not None:
    lw = result.weights.iloc[-1]
    weights = lw[lw.abs() > 1e-6]
    src_label = "latest backtest weights"
up = st.file_uploader("…or upload weights CSV (ticker, weight)", type=["csv"])
if up is not None:
    wdf = pd.read_csv(up)
    tcol, wcol = wdf.columns[0], wdf.columns[1]
    weights = pd.Series(pd.to_numeric(wdf[wcol], errors="coerce").values,
                        index=wdf[tcol].astype(str).str.upper()).dropna()
    src_label = f"uploaded ({up.name})"

if weights is None or weights.empty:
    st.info("Run a backtest on the Backtest Lab page, or upload a weights CSV.")
    st.stop()

c1, c2, c3 = st.columns(3)
aum = c1.number_input("AUM ($M)", 1.0, 100000.0, 100.0, 10.0) * 1e6
c2.metric("Positions", f"{(weights.abs() > 1e-6).sum()}",
          f"gross {weights.abs().sum():.0%} · net {weights.sum():+.0%}",
          delta_color="off")
c3.caption(f"Book source: **{src_label}**")

factors = _factors()
if factors.empty:
    st.error("No factor data available from the active source.")
    st.stop()
factor_cols = [c for c in factors.columns if str(c).upper() != "RF"]

# ---------------- Exposures ----------------
prices = _source().get_price_panel(list(weights.index))
wdf_panel = pd.DataFrame([weights.reindex(prices.columns).fillna(0.0)],
                         index=[prices.index[-1] if not prices.empty
                                else pd.Timestamp.now().normalize()])

exposure = pd.DataFrame()
used_live = False
if HAVE_FIRM_UTILS:
    try:
        from quantdash.engine.factors import live_zscored_exposure_heatmap_data

        exposure = live_zscored_exposure_heatmap_data(
            trading_tickers=list(weights.index),
            notionals=weights * aum, history_days=30)
        used_live = not exposure.empty
    except Exception as e:
        st.warning(f"Live Axioma exposures unavailable ({e}) — falling back to "
                   "regression betas.")
if exposure.empty and not prices.empty:
    beta_contrib = factor_exposure_heatmap_data(wdf_panel, prices, factors)
    if not beta_contrib.empty:
        exposure = beta_contrib * aum  # weight x beta x AUM = factor $ exposure

if exposure.empty:
    st.error("Could not compute exposures for this book.")
    st.stop()

port_exposure = exposure.sum()  # $ exposure per factor
st.subheader("Portfolio factor exposure "
             + ("(live Axioma z-scored)" if used_live
                else "(trailing regression betas)"))
srt = port_exposure.sort_values()
fig_e = go.Figure(go.Bar(
    x=srt.values / 1e6, y=srt.index, orientation="h",
    marker=dict(color=diverging_colors(srt.values), line=dict(width=0)),
    text=[f"{v/1e6:+,.1f}M" for v in srt.values], textposition="outside",
    cliponaxis=False))
style_fig(fig_e, height=32 * len(srt) + 90, hover="closest", show_legend=False,
          title="Net factor dollar exposure")
fig_e.update_xaxes(title="$M")
st.plotly_chart(fig_e, width="stretch")

# Per-name heatmap
fig_h = go.Figure(go.Heatmap(
    z=exposure.values / 1e6, x=list(exposure.columns), y=list(exposure.index),
    colorscale=[[0, "rgba(240,84,79,0.9)"], [0.5, "rgba(21,27,43,1)"],
                [1, "rgba(46,194,126,0.9)"]], zmid=0,
    colorbar=dict(thickness=10, outlinewidth=0, title="$M"),
    hovertemplate="%{y} · %{x}: $%{z:.2f}M<extra></extra>"))
style_fig(fig_h, height=max(320, 20 * len(exposure)), hover="closest",
          show_legend=False, title="Per-name factor dollar exposure ($M)")
st.plotly_chart(fig_h, width="stretch")

# ---------------- Stress tests ----------------
st.subheader("Factor stress tests")
fac_daily = factors[[c for c in factor_cols if c in port_exposure.index]]
sigma = fac_daily.std()

shock_z = st.slider("Shock size (σ, daily)", 1.0, 4.0, 2.0, 0.5)
rows = []
for f in sigma.index:
    move = shock_z * sigma[f]
    pnl = float(port_exposure[f] * move)
    rows.append({"factor": f, "shock": f"±{shock_z:.0f}σ = ±{move:.2%}",
                 "pnl_up": pnl, "pnl_down": -pnl})
stress = pd.DataFrame(rows).set_index("factor")
stress["abs"] = stress["pnl_up"].abs()
stress = stress.sort_values("abs", ascending=False).drop(columns="abs")

sc1, sc2 = st.columns([2, 3])
sc1.dataframe(
    stress.style.format({"pnl_up": "${:+,.0f}", "pnl_down": "${:+,.0f}"}),
    width="stretch")
worst = stress["pnl_up"].abs().idxmax()
sc2.markdown(
    f"**Biggest single-factor vulnerability: {worst}** — a {shock_z:.0f}σ move "
    f"is ±${stress.loc[worst, 'pnl_up']:+,.0f} on ${aum/1e6:,.0f}M "
    f"({stress.loc[worst, 'pnl_up']/aum:+.2%} of NAV).\n\n"
    "Signs matter: exposures net across longs and shorts, so a hedged book "
    "should show small bars here even with large gross positions.")

# Historical worst factor days for this book
port_factor_pnl = (fac_daily * port_exposure.reindex(fac_daily.columns)).sum(axis=1)
worst_days = port_factor_pnl.nsmallest(5)
best_days = port_factor_pnl.nlargest(5)
hd = pd.DataFrame({
    "date": [d.date() for d in worst_days.index] + [d.date() for d in best_days.index],
    "factor P&L": list(worst_days.values) + list(best_days.values),
    "type": ["worst"] * 5 + ["best"] * 5,
})
st.subheader("Historical scenario replay")
st.dataframe(
    hd.style.format({"factor P&L": "${:+,.0f}"})
    .map(lambda v: "color: crimson" if v == "worst"
         else ("color: seagreen" if v == "best" else ""), subset=["type"]),
    width="stretch", hide_index=True)
st.caption("Today's factor exposures replayed through every historical factor "
           "day — the factor-driven P&L this book would have had, ignoring "
           "idiosyncratic moves.")
