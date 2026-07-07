"""Theory Journal — saved hypotheses with their test results, side by side."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantdash.data import get_source
from quantdash.ui import GREEN, inject_css, style_fig

st.set_page_config(page_title="Theory Journal", page_icon="📓", layout="wide")
inject_css()
st.title("📓 Theory Journal")
st.caption("Every hypothesis you've tested, with the numbers that confirmed or "
           "killed it. Save runs from the Backtest Lab.")


@st.cache_resource
def _source():
    return get_source()


src = _source()
theories = src.list_theories()

if theories.empty:
    st.info("No theories saved yet. Run a backtest on the Home page and save it.")
    st.stop()

rows = []
for _, t in theories.iterrows():
    m = json.loads(t["metrics_json"] or "{}")
    rows.append({
        "name": t["name"],
        "verdict": t["verdict"],
        "sharpe": m.get("sharpe"),
        "lo_sharpe": m.get("sharpe_lo_corrected"),
        "p": m.get("sharpe_p_value"),
        "ann_ret": m.get("ann_return"),
        "max_dd": m.get("max_drawdown"),
        "capm_α": m.get("capm_alpha_ann"),
        "ic_t": m.get("ic_tstat"),
        "turnover": m.get("ann_turnover"),
        "expression": t["expression"],
        "created": str(t["created_at"])[:16],
        "id": t["id"],
    })
df = pd.DataFrame(rows)

verdict_emoji = {"supported": "✅", "not supported": "❌", "inconclusive": "❔"}
df["verdict"] = df["verdict"].map(lambda v: f"{verdict_emoji.get(v, '')} {v}")

st.dataframe(
    df.drop(columns=["id"]).style.format({
        "sharpe": "{:.2f}", "lo_sharpe": "{:.2f}", "p": "{:.3f}",
        "ann_ret": "{:.1%}", "max_dd": "{:.1%}", "capm_α": "{:.1%}",
        "ic_t": "{:.2f}", "turnover": "{:.0%}"}, na_rep="—"),
    use_container_width=True, height=min(600, 60 + 36 * len(df)),
)

# Sharpe comparison chart
fig = go.Figure(go.Bar(
    x=df["name"], y=df["sharpe"],
    marker=dict(color=["rgba(46,194,126,0.85)" if (p or 1) < 0.05
                       else "rgba(139,147,167,0.45)" for p in df["p"]],
                line=dict(width=0)),
    text=[f"{s:.2f}" if s is not None else "" for s in df["sharpe"]],
    textposition="outside", textfont=dict(size=11), cliponaxis=False,
))
style_fig(fig, height=340, hover="closest", show_legend=False,
          title="Sharpe by theory (green = p < 0.05, Lo-corrected)")
fig.update_layout(bargap=0.4)
st.plotly_chart(fig, use_container_width=True)

# Detail / manage
st.divider()
sel = st.selectbox("Inspect a theory", df["name"] + "  ·  " + df["id"])
tid = sel.split("·")[-1].strip()
t = theories[theories["id"] == tid].iloc[0]
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Hypothesis:** {t['hypothesis'] or '_none recorded_'}")
    st.code(t["expression"], language="python")
    st.json(json.loads(t["config_json"] or "{}"), expanded=False)
with c2:
    st.json(json.loads(t["metrics_json"] or "{}"), expanded=False)
    if st.button("🗑 Delete this theory", type="secondary"):
        src.delete_theory(tid)
        st.rerun()
    st.caption("To re-run it, copy the expression into the Backtest Lab.")
