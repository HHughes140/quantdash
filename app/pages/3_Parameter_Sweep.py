"""Parameter Sweep — test a theory's robustness across parameter values.

A signal that only works at one exact lookback is noise; a real effect shows a
plateau. Write an expression with {A} (and optionally {B}) placeholders and
sweep a grid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantdash.data import DuckDBSource, get_source
from quantdash.engine import (BacktestConfig, compute_metrics, evaluate_signal,
                              run_backtest)
from quantdash.data.universe import BENCHMARKS
from quantdash.workspace import load_workspace
from quantdash.ui import ACCENT, GREEN, RED, diverging_colors, inject_css, page_header, style_fig

st.set_page_config(page_title="Parameter Sweep — Insurance Alpha Lab",
                   page_icon="🎛️", layout="wide")
inject_css()
page_header("Parameter Sweep", "Robustness across the parameter grid")
st.caption("A real effect is robust to its parameters — look for a plateau, "
           "not a spike. Metrics below are computed on the OOS segment to keep "
           "the sweep honest.")

MAX_COMBOS = 64


@st.cache_resource
def _source():
    return get_source()


def _defs():
    return load_workspace().get("definitions") or {}


@st.cache_data(ttl=3600, show_spinner=False)
def _macro():
    try:
        return DuckDBSource().get_macro_panel()
    except Exception:
        import pandas as _pd
        return _pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _panels():
    src = _source()
    tickers = [t for t in src.available_tickers() if t not in BENCHMARKS]
    prices = src.get_price_panel(tickers)
    volume = src.get_price_panel(tickers, field="volume")
    bench = src.get_price_panel(["SPY"])
    keep = prices.columns[prices.notna().mean() > 0.6]
    return prices[keep], volume.reindex(columns=keep), \
        (bench["SPY"] if not bench.empty else None)


if not _source().available_tickers():
    st.warning("No price data — seed it in Data Explorer first.")
    st.stop()

# ---------------- Controls ----------------
c1, c2 = st.columns([2, 1])
template = c1.text_area(
    "Signal template — use {A} and optionally {B} as placeholders",
    value="rank(momentum({A}, 21))", height=70)
metric_name = c2.selectbox(
    "Objective", ["OOS Sharpe", "Full-sample Sharpe", "OOS IC t-stat",
                  "OOS ann. return", "OOS max drawdown"])

c3, c4, c5 = st.columns(3)
a_vals_raw = c3.text_input("A values (comma-separated)", "21, 63, 126, 189, 252")
b_vals_raw = c4.text_input("B values (optional)", "")
oos_frac = c5.slider("OOS fraction", 0.1, 0.5, 0.3, 0.05)

c6, c7, c8, c9 = st.columns(4)
mode = c6.selectbox("Construction", ["long_short", "long_only", "signal_weight"])
quantile = c7.slider("Quantile", 0.05, 0.5, 0.2, 0.05)
rebal = c8.selectbox("Rebalance (days)", [1, 5, 10, 21, 63], index=1)
cost_bps = c9.number_input("Cost (bps)", 0.0, 100.0, 5.0, 1.0)


def _parse(raw: str) -> list:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            try:
                out.append(float(tok))
            except ValueError:
                out.append(tok)
    return out


def _objective(res, m_oos, m_full):
    return {
        "OOS Sharpe": m_oos.get("sharpe"),
        "Full-sample Sharpe": m_full.get("sharpe"),
        "OOS IC t-stat": m_oos.get("ic_tstat"),
        "OOS ann. return": m_oos.get("ann_return"),
        "OOS max drawdown": m_oos.get("max_drawdown"),
    }[metric_name]


if st.button("▶ Run sweep", type="primary"):
    a_vals, b_vals = _parse(a_vals_raw), _parse(b_vals_raw)
    if not a_vals:
        st.error("Give at least one A value.")
        st.stop()
    combos = [(a, b) for a in a_vals for b in (b_vals or [None])]
    if len(combos) > MAX_COMBOS:
        st.error(f"{len(combos)} combinations — cap is {MAX_COMBOS}. Trim the grid.")
        st.stop()

    prices, volume, bench = _panels()
    cfg = BacktestConfig(mode=mode, quantile=quantile, rebalance_every=int(rebal),
                         cost_bps=cost_bps)
    rows, prog = [], st.progress(0.0, text="Sweeping...")
    for i, (a, b) in enumerate(combos):
        expr = template.replace("{A}", str(a))
        if b is not None:
            expr = expr.replace("{B}", str(b))
        label = f"A={a}" + (f", B={b}" if b is not None else "")
        try:
            sig = evaluate_signal(expr, prices, volume, _macro(),
                                  definitions=_defs())
            res = run_backtest(prices, sig, cfg, bench)
            ridx = res.net_returns.index
            split = ridx[int(len(ridx) * (1 - oos_frac))]
            ic_oos = res.ic.loc[split:]
            m_full = compute_metrics(res.net_returns, ic=res.ic)
            m_oos = compute_metrics(res.net_returns.loc[split:], ic=ic_oos)
            if "error" in m_full or "error" in m_oos:
                raise ValueError("not enough observations")
            rows.append({"A": a, "B": b, "label": label,
                         "objective": _objective(res, m_oos, m_full),
                         "IS Sharpe": compute_metrics(
                             res.net_returns.loc[:split]).get("sharpe"),
                         "OOS Sharpe": m_oos.get("sharpe"),
                         "Full Sharpe": m_full.get("sharpe"),
                         "OOS ann ret": m_oos.get("ann_return"),
                         "OOS max DD": m_oos.get("max_drawdown"),
                         "OOS IC t": m_oos.get("ic_tstat"),
                         "turnover": m_full.get("ann_turnover"),
                         "error": None})
        except ValueError as e:
            rows.append({"A": a, "B": b, "label": label, "objective": np.nan,
                         "error": str(e)})
        prog.progress((i + 1) / len(combos), text=f"Sweeping... {label}")
    prog.empty()
    st.session_state["sweep_df"] = pd.DataFrame(rows)
    st.session_state["sweep_meta"] = {"template": template, "metric": metric_name,
                                      "has_b": bool(b_vals)}

df = st.session_state.get("sweep_df")
if df is None:
    st.info("Define a template and grid, then run the sweep.")
    st.stop()
meta = st.session_state["sweep_meta"]

errs = df[df["error"].notna()]
if not errs.empty:
    st.warning(f"{len(errs)} combination(s) failed: "
               + "; ".join(f"{r['label']} ({r['error'][:60]})"
                           for _, r in errs.head(3).iterrows()))
ok = df[df["error"].isna()].copy()
if ok.empty:
    st.error("Every combination failed — check the template syntax.")
    st.stop()

best = ok.loc[ok["objective"].idxmax()]
st.metric(f"Best {meta['metric']}", f"{best['objective']:.3f}",
          best["label"], delta_color="off")

if meta["has_b"]:
    piv = ok.pivot(index="B", columns="A", values="objective")
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[str(c) for c in piv.columns],
        y=[str(i) for i in piv.index],
        colorscale=[[0, "rgba(240,84,79,0.9)"], [0.5, "rgba(21,27,43,1)"],
                    [1, "rgba(46,194,126,0.9)"]], zmid=0,
        text=np.round(piv.values, 2), texttemplate="%{text}",
        colorbar=dict(thickness=10, outlinewidth=0),
        hovertemplate="A=%{x}, B=%{y}: %{z:.3f}<extra></extra>"))
    style_fig(fig, height=140 + 40 * len(piv), hover="closest", show_legend=False,
              title=f"{meta['metric']} across the grid")
    fig.update_xaxes(title="A", type="category")
    fig.update_yaxes(title="B", type="category")
    st.plotly_chart(fig, width="stretch")
else:
    fig = go.Figure(go.Bar(
        x=[str(a) for a in ok["A"]], y=ok["objective"],
        marker=dict(color=diverging_colors(ok["objective"].fillna(0)),
                    line=dict(width=0)),
        text=[f"{v:.2f}" for v in ok["objective"]], textposition="outside",
        cliponaxis=False))
    style_fig(fig, height=340, hover="closest", show_legend=False,
              title=f"{meta['metric']} by A")
    fig.update_layout(bargap=0.4)
    fig.update_xaxes(title="A", type="category")
    st.plotly_chart(fig, width="stretch")

# IS vs OOS scatter — the overfitting picture
fig_sc = go.Figure()
fig_sc.add_trace(go.Scatter(
    x=ok["IS Sharpe"], y=ok["OOS Sharpe"], mode="markers+text",
    text=ok["label"], textposition="top center", textfont=dict(size=10),
    marker=dict(size=10, color=ACCENT, line=dict(width=1, color="#2A3247")),
    hovertemplate="%{text}<br>IS %{x:.2f} · OOS %{y:.2f}<extra></extra>"))
lims = [min(ok[["IS Sharpe", "OOS Sharpe"]].min().min(), 0) - 0.2,
        ok[["IS Sharpe", "OOS Sharpe"]].max().max() + 0.2]
fig_sc.add_trace(go.Scatter(x=lims, y=lims, mode="lines", showlegend=False,
                            line=dict(color="rgba(255,255,255,0.2)", dash="dot")))
style_fig(fig_sc, height=380, hover="closest", show_legend=False,
          title="In-sample vs out-of-sample Sharpe (points below the line "
                "degrade OOS)")
fig_sc.update_xaxes(title="IS Sharpe")
fig_sc.update_yaxes(title="OOS Sharpe")
st.plotly_chart(fig_sc, width="stretch")

show_cols = [c for c in ["label", "IS Sharpe", "OOS Sharpe", "Full Sharpe",
                         "OOS ann ret", "OOS max DD", "OOS IC t", "turnover"]
             if c in ok.columns]
st.dataframe(
    ok[show_cols].sort_values("OOS Sharpe", ascending=False).style.format({
        "IS Sharpe": "{:.2f}", "OOS Sharpe": "{:.2f}", "Full Sharpe": "{:.2f}",
        "OOS ann ret": "{:.1%}", "OOS max DD": "{:.1%}", "OOS IC t": "{:.2f}",
        "turnover": "{:.0%}"}, na_rep="—"),
    width="stretch", hide_index=True)
st.caption("Robustness read: adjacent parameter values should have similar OOS "
           "numbers. One spike surrounded by noise = you found luck, not signal.")
