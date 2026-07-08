"""Walk-Forward — re-select the best parameter each window, trade it forward.

The honest version of a parameter sweep: at each step the parameter is chosen
using ONLY data available at the time, then traded out-of-sample. The stitched
OOS curve is what a disciplined process would actually have earned.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantdash.data import DuckDBSource, get_source
from quantdash.data.universe import BENCHMARKS
from quantdash.engine import BacktestConfig, compute_metrics
from quantdash.engine.walkforward import candidate_returns, walk_forward
from quantdash.workspace import load_workspace
from quantdash.ui import (ACCENT, GOLD, GRAY, GREEN, RED, diverging_colors,
                          inject_css, page_header, style_fig)

st.set_page_config(page_title="Walk-Forward — Insurance Alpha Lab",
                   page_icon="🚶", layout="wide")
inject_css()
page_header("Walk-Forward",
            "Parameter chosen only with information available at the time")


@st.cache_resource
def _source():
    return get_source()


@st.cache_data(ttl=3600, show_spinner=False)
def _macro():
    try:
        return DuckDBSource().get_macro_panel()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _panels():
    src = _source()
    tickers = [t for t in src.available_tickers() if t not in BENCHMARKS]
    prices = src.get_price_panel(tickers)
    volume = src.get_price_panel(tickers, field="volume")
    bench = src.get_price_panel(["SPY"])
    keep = prices.columns[prices.notna().mean() > 0.6]
    return prices[keep], volume.reindex(columns=keep), \
        (bench["SPY"].pct_change() if not bench.empty else None)


if not _source().available_tickers():
    st.warning("No price data — seed it in Data Explorer first.")
    st.stop()

# ---------------- Controls ----------------
c1, c2 = st.columns([2, 1])
template = c1.text_area("Signal template with {A} placeholder",
                        value="rank(momentum({A}, 21))", height=70)
a_vals_raw = c2.text_input("A candidates", "63, 126, 189, 252")

c3, c4, c5, c6 = st.columns(4)
train_yrs = c3.selectbox("Train window", ["1y", "2y", "3y"], index=1)
test_m = c4.selectbox("Test window", ["3m", "6m", "12m"], index=1)
mode = c5.selectbox("Construction", ["long_short", "long_only", "signal_weight"])
cost_bps = c6.number_input("Cost (bps)", 0.0, 100.0, 5.0, 1.0)
c7, c8 = st.columns(4)[:2]
quantile = c7.slider("Quantile", 0.05, 0.5, 0.2, 0.05)
rebal = c8.selectbox("Rebalance (days)", [1, 5, 10, 21], index=1)

TRAIN = {"1y": 252, "2y": 504, "3y": 756}[train_yrs]
TEST = {"3m": 63, "6m": 126, "12m": 252}[test_m]

if st.button("▶ Run walk-forward", type="primary"):
    a_vals = [v.strip() for v in a_vals_raw.split(",") if v.strip()]
    a_vals = [int(v) if v.isdigit() else v for v in a_vals]
    if not a_vals:
        st.error("Give at least one candidate value.")
        st.stop()
    if "{A}" not in template:
        st.error("Template needs an {A} placeholder.")
        st.stop()

    prices, volume, bench = _panels()
    cfg = BacktestConfig(mode=mode, quantile=quantile,
                         rebalance_every=int(rebal), cost_bps=cost_bps)
    prog = st.progress(0.0, text="Backtesting candidates...")
    try:
        cand = candidate_returns(
            prices, volume, template, a_vals, cfg, _macro(),
            progress=lambda f, a: prog.progress(
                f * 0.9, text=f"Backtesting candidates... A={a}"),
            definitions=load_workspace().get("definitions") or {})
        prog.progress(0.95, text="Stitching walk-forward windows...")
        wf, windows = walk_forward(cand, train_days=TRAIN, test_days=TEST)
    except ValueError as e:
        prog.empty()
        st.error(str(e))
        st.stop()
    prog.empty()
    st.session_state["wf"] = {"wf": wf, "windows": windows, "cand": cand,
                              "bench": bench, "template": template,
                              "train": train_yrs, "test": test_m}

state = st.session_state.get("wf")
if state is None:
    st.info("Define a template and candidate grid, then run.")
    st.stop()

wf, windows, cand = state["wf"], state["windows"], state["cand"]
bench = state["bench"]

# ---------------- Results ----------------
m = compute_metrics(wf)
static_best = cand.loc[wf.index].apply(
    lambda c: c.mean() / c.std() * np.sqrt(252) if c.std() > 0 else np.nan)
row = st.columns(6)
row[0].metric("WF Sharpe (OOS)", f"{m['sharpe']:.2f}")
row[1].metric("Lo-corrected", f"{m['sharpe_lo_corrected']:.2f}",
              f"p={m['sharpe_p_value']:.3f}", delta_color="off")
row[2].metric("Ann. return", f"{m['ann_return']:.1%}")
row[3].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
row[4].metric("Windows positive", f"{(windows['oos_return'] > 0).mean():.0%}",
              f"{len(windows)} windows", delta_color="off")
row[5].metric("Best static (hindsight)", f"{static_best.max():.2f}",
              f"A={static_best.idxmax()}", delta_color="off")
gap = m["sharpe"] - static_best.max()
st.caption(
    f"Walk-forward vs best-in-hindsight static gap: **{gap:+.2f}** Sharpe. "
    "A small gap means the process finds the right parameter in real time; a "
    "large one means the sweep's best number was hindsight. Parameter switches "
    "at window boundaries are assumed costless (slightly optimistic).")

# Equity: WF stitched vs each static candidate vs benchmark
fig = go.Figure()
for a in cand.columns:
    eq_a = (1 + cand[a].loc[wf.index].fillna(0)).cumprod()
    fig.add_trace(go.Scatter(x=eq_a.index, y=eq_a, name=f"static A={a}",
                             line=dict(width=1, color=GRAY), opacity=0.45,
                             hovertemplate="%{y:.2f}<extra>A=" + str(a) + "</extra>"))
if bench is not None:
    beq = (1 + bench.reindex(wf.index).fillna(0)).cumprod()
    fig.add_trace(go.Scatter(x=beq.index, y=beq, name="SPY",
                             line=dict(width=1.2, color="#6B7280", dash="dot")))
eq = (1 + wf.fillna(0)).cumprod()
fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Walk-forward (stitched OOS)",
                         line=dict(width=2.6, color=ACCENT),
                         hovertemplate="%{y:.2f}<extra>walk-forward</extra>"))
for _, wrow in windows.iterrows():
    fig.add_vline(x=wrow["window_start"], line_width=1,
                  line_color="rgba(255,255,255,0.08)")
style_fig(fig, height=460,
          title=f"Stitched OOS equity — train {state['train']}, "
                f"test {state['test']} (gray = static candidates)")
st.plotly_chart(fig, width="stretch")

# Chosen parameter over time + per-window Sharpes
w1, w2 = st.columns(2)
fig_p = go.Figure(go.Scatter(
    x=windows["window_start"], y=[str(p) for p in windows["chosen_param"]],
    mode="lines+markers", line=dict(shape="hv", color=GOLD, width=2),
    marker=dict(size=8),
    hovertemplate="%{x|%Y-%m}: A=%{y}<extra></extra>"))
style_fig(fig_p, height=300, hover="closest", show_legend=False,
          title="Chosen parameter per window")
fig_p.update_yaxes(type="category", title="A")
w1.plotly_chart(fig_p, width="stretch")
st.caption("A stable chosen parameter = robust signal. Thrashing between "
           "extremes = the training window is fitting noise.")

fig_w = go.Figure()
fig_w.add_trace(go.Bar(x=windows["window_start"], y=windows["oos_sharpe"],
                       name="OOS (chosen)",
                       marker=dict(color=diverging_colors(
                           windows["oos_sharpe"].fillna(0)), line=dict(width=0))))
fig_w.add_trace(go.Scatter(x=windows["window_start"],
                           y=windows["best_hindsight_sharpe"],
                           name="best hindsight", mode="markers",
                           marker=dict(symbol="line-ew-open", size=14,
                                       color="#E6E9F0")))
style_fig(fig_w, height=300, hover="closest",
          title="Per-window OOS Sharpe vs best hindsight")
w2.plotly_chart(fig_w, width="stretch")

st.dataframe(
    windows.assign(
        window_start=windows["window_start"].dt.date,
        window_end=windows["window_end"].dt.date)
    .style.format({"train_sharpe": "{:.2f}", "oos_sharpe": "{:.2f}",
                   "oos_return": "{:.1%}", "best_hindsight_sharpe": "{:.2f}"},
                  na_rep="—"),
    width="stretch", hide_index=True)

if st.button("➕ Add stitched WF to Compare (Backtest Lab tab)"):
    snap = {k: v for k, v in m.items() if not isinstance(v, (tuple, str))}
    st.session_state.setdefault("compare", {})[
        f"WF {state['template'][:30]}"] = {
        "returns": wf, "metrics": snap,
        "expression": f"[walk-forward] {state['template']}",
        "config": {"train": state["train"], "test": state["test"]},
    }
    st.success("Added — open the Compare tab on the Backtest Lab page.")
