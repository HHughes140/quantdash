"""Shared visual theme: plotly styling, factor colors, and app CSS."""

from __future__ import annotations

import plotly.graph_objects as go

# Core palette (dark navy UI)
ACCENT = "#5B8DEF"   # strategy / primary
GREEN = "#2EC27E"
RED = "#F0544F"
GOLD = "#E5B454"
PURPLE = "#9B7EDE"
CYAN = "#4FC3F7"
GRAY = "#8B93A7"
BG_CARD = "#151B2B"
GRID = "rgba(255,255,255,0.07)"
AXIS = "#2A3247"

FACTOR_COLORS = {
    "MKT_RF": GRAY,
    "SMB": CYAN,
    "HML": GOLD,
    "RMW": PURPLE,
    "CMA": "#6FBF9B",
    "MOM": RED,
    "Risk-free": "#5A6478",
    "Residual (alpha)": GREEN,
}


def style_fig(
    fig: go.Figure,
    height: int = 400,
    title: str | None = None,
    ytickformat: str | None = None,
    hover: str = "x unified",
    show_legend: bool = True,
) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, Inter, 'Segoe UI', sans-serif",
                  size=12, color="#C9D1E3"),
        height=height,
        margin=dict(t=48 if title else 20, b=12, l=10, r=10),
        hovermode=hover,
        hoverlabel=dict(bgcolor=BG_CARD, bordercolor=AXIS, font_size=12),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left",
                                     font=dict(size=14, color="#E6E9F0")))
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True,
                     linecolor=AXIS, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, zeroline=True,
                     zerolinecolor="rgba(255,255,255,0.18)", zerolinewidth=1,
                     showline=False)
    if ytickformat:
        fig.update_yaxes(tickformat=ytickformat)
    return fig


def diverging_colors(values, pos=GREEN, neg=RED) -> list[str]:
    """Solid green/red with opacity scaled by magnitude — reads well on dark bg."""
    vals = list(values)
    vmax = max((abs(v) for v in vals), default=1) or 1
    out = []
    for v in vals:
        a = 0.35 + 0.65 * min(abs(v) / vmax, 1.0)
        base = pos if v >= 0 else neg
        r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
        out.append(f"rgba({r},{g},{b},{a:.2f})")
    return out


CSS = """
<style>
/* metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(91,141,239,0.07), rgba(21,27,43,0.55));
    border: 1px solid #232B40;
    border-radius: 12px;
    padding: 12px 16px 10px 16px;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.70rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #8B93A7 !important;
}
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }
[data-testid="stMetricDelta"] { font-size: 0.78rem; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #1C2333; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px 8px 0 0;
    padding: 8px 16px; color: #8B93A7;
}
.stTabs [aria-selected="true"] {
    background: #1B2438; color: #E6E9F0;
    border-bottom: 2px solid #5B8DEF;
}

/* sidebar + headers */
[data-testid="stSidebar"] { border-right: 1px solid #1C2333; }
h1, h2, h3 { letter-spacing: -0.01em; }
hr { border-color: #1C2333; }

/* dataframes */
[data-testid="stDataFrame"] { border: 1px solid #232B40; border-radius: 10px; }
</style>
"""


def inject_css() -> None:
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)
