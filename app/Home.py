"""Backtest Lab — main page. Run: streamlit run app/Home.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from quantdash.data import get_source, get_theory_store
from quantdash.data.universe import BENCHMARKS, INSURANCE_UNIVERSE, SUBSECTOR
from quantdash.engine import (
    SIGNAL_PRESETS,
    BacktestConfig,
    compute_metrics,
    evaluate_signal,
    factor_regression,
    rolling_factor_betas,
    run_backtest,
)
from quantdash.engine.attribution import (
    best_worst_windows,
    factor_contribution,
    position_contribution,
    universe_performance,
)
from quantdash.engine.factors import factor_exposure_heatmap_data
from quantdash.engine.metrics import drawdown_series, monthly_return_table
from quantdash.ui import (
    ACCENT, CYAN, FACTOR_COLORS, GOLD, GRAY, GREEN, PURPLE, RED,
    SUBSECTOR_COLORS, diverging_colors, factor_color, inject_css, page_header,
    style_fig, with_alpha,
)

st.set_page_config(page_title="Insurance Alpha Lab", page_icon="🏛️", layout="wide")
inject_css()


@st.cache_resource
def _source():
    return get_source()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_panel(tickers: tuple, start: str, end: str, field: str) -> pd.DataFrame:
    return _source().get_price_panel(list(tickers), start, end, field)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_factors(start: str, end: str) -> pd.DataFrame:
    return _source().get_factors(start=start, end=end)


@st.cache_resource
def _theories():
    return get_theory_store(_source())


src = _source()
_SOURCE_BADGES = {
    "snowflake_axioma_read_only": "Axioma WW4 · Snowflake",
    "snowflake": "Snowflake",
    "duckdb": "Local cache",
}
page_header("Insurance Alpha Lab",
            "Signal research · factor overlays · theory testing",
            badge=_SOURCE_BADGES.get(src.name, src.name))

tickers_all = src.available_tickers()
if not tickers_all:
    st.warning(
        "No price data yet. Go to **Data Explorer** (sidebar) and seed the "
        "database, or run `python scripts/seed_local.py`."
    )
    st.stop()

# ---------------- Sidebar: experiment definition ----------------
with st.sidebar:
    st.header("Experiment")

    insurance_avail = [t for t in tickers_all if t in SUBSECTOR]
    universe_opts = (["Insurance", "Insurance subsectors"] if insurance_avail else []) \
        + ["All available", "Custom"]
    universe_mode = st.radio("Universe", universe_opts, horizontal=True)
    if universe_mode == "Insurance":
        tickers = insurance_avail
        st.caption(f"{len(tickers)} insurance names across "
                   f"{len({SUBSECTOR[t] for t in tickers})} subsectors")
    elif universe_mode == "Insurance subsectors":
        subs = st.multiselect("Subsectors", list(INSURANCE_UNIVERSE),
                              default=list(INSURANCE_UNIVERSE)[:4])
        tickers = [t for t in insurance_avail if SUBSECTOR[t] in subs]
    elif universe_mode == "Custom":
        tickers = st.multiselect("Tickers", tickers_all,
                                 default=tickers_all[: min(50, len(tickers_all))])
    else:
        tickers = [t for t in tickers_all if t not in BENCHMARKS]

    bench_avail = [b for b in BENCHMARKS if b in tickers_all]
    if bench_avail:
        default_bench = ("KIE" if "KIE" in bench_avail
                         and universe_mode.startswith("Insurance") else bench_avail[0])
        bench_ticker = st.selectbox("Benchmark", bench_avail,
                                    index=bench_avail.index(default_bench))
    else:
        bench_ticker = None

    cov = src.coverage()
    dmin, dmax = pd.to_datetime(cov["start"].min()), pd.to_datetime(cov["end"].max())
    start, end = st.slider(
        "Date range",
        min_value=dmin.to_pydatetime(), max_value=dmax.to_pydatetime(),
        value=(dmin.to_pydatetime(), dmax.to_pydatetime()), format="YYYY-MM",
    )

    st.subheader("Signal")
    preset = st.selectbox("Preset", ["— custom —"] + list(SIGNAL_PRESETS))
    default_expr = SIGNAL_PRESETS.get(preset, st.session_state.get(
        "expression", "rank(momentum(252, 21))"))
    expression = st.text_area("Expression (higher = more attractive)",
                              value=default_expr, height=80)
    with st.expander("DSL reference"):
        st.markdown(
            "**Time-series:** `returns(n)`, `momentum(lb, skip)`, `vol(n)`, "
            "`sma(n)`, `ema(n)`, `price()`, `drawdown(n)`, `rsi(n)`, "
            "`volume_ratio(s, l)`, `delay(x, n)`, `delta(x, n)`, "
            "`ts_rank(x, n)`, `ts_zscore(x, n)`\n\n"
            "**Cross-sectional:** `rank(x)`, `zscore(x)`, `demean(x)`, "
            "`winsorize(x, z)`\n\n"
            "**Math:** `log sqrt abs sign exp clip(x, lo, hi) where(cond, a, b)`"
        )

    st.subheader("Portfolio")
    mode = st.selectbox("Construction", ["long_short", "long_only", "signal_weight"],
                        format_func=lambda x: {
                            "long_short": "Long/short quantiles (dollar-neutral)",
                            "long_only": "Long-only top quantile",
                            "signal_weight": "Signal-proportional weights"}[x])
    c1, c2 = st.columns(2)
    quantile = c1.slider("Quantile", 0.05, 0.5, 0.2, 0.05)
    rebal = c2.selectbox("Rebalance (days)", [1, 5, 10, 21, 63], index=1)
    c3, c4 = st.columns(2)
    cost_bps = c3.number_input("Cost (bps, one-way)", 0.0, 100.0, 5.0, 1.0)
    max_w = c4.number_input("Max weight", 0.01, 1.0, 0.10, 0.01)
    vol_target = st.number_input("Vol target (0 = off)", 0.0, 0.5, 0.0, 0.01)
    oos_frac = st.slider("Hold-out (OOS) fraction", 0.0, 0.5, 0.3, 0.05,
                         help="Last X% of the period is treated as out-of-sample; "
                              "metrics are reported separately so you can see if "
                              "the signal survives outside the fitting window.")

    run = st.button("▶ Run backtest", type="primary", use_container_width=True)

# ---------------- Run ----------------
if run:
    with st.spinner("Loading data and running backtest..."):
        s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        prices = _load_panel(tuple(sorted(tickers)), s, e, "adj_close")
        volume = _load_panel(tuple(sorted(tickers)), s, e, "volume")
        bench = (_load_panel((bench_ticker,), s, e, "adj_close")
                 if bench_ticker else pd.DataFrame())
        factors = _load_factors(s, e)

        if prices.empty:
            st.error("No prices for that selection.")
            st.stop()
        # Drop names with sparse history (<60% of days)
        keep = prices.columns[prices.notna().mean() > 0.6]
        prices, volume = prices[keep], volume.reindex(columns=keep)

        try:
            signal = evaluate_signal(expression, prices, volume)
        except ValueError as err:
            st.error(str(err))
            st.stop()

        cfg = BacktestConfig(
            mode=mode, quantile=quantile, rebalance_every=int(rebal),
            cost_bps=cost_bps, max_weight=max_w,
            vol_target=vol_target or None,
        )
        try:
            result = run_backtest(prices, signal, cfg,
                                  bench[bench_ticker] if not bench.empty else None)
        except ValueError as err:
            st.error(str(err))
            st.stop()

        st.session_state.update(
            result=result, prices=prices, factors=factors,
            expression=expression, cfg=cfg, signal=signal, oos_frac=oos_frac,
            bench_ticker=bench_ticker if not bench.empty else None,
        )

if "result" not in st.session_state:
    st.info("Define a signal in the sidebar and hit **Run backtest**.")
    st.stop()

result = st.session_state["result"]
prices = st.session_state["prices"]
factors = st.session_state["factors"]
cfg = st.session_state["cfg"]
bench_label = st.session_state.get("bench_ticker") or "benchmark"

gross_ann = (1 + result.gross_returns.dropna()).prod() ** (252 / len(result.gross_returns.dropna())) - 1
net_ann = (1 + result.net_returns.dropna()).prod() ** (252 / len(result.net_returns.dropna())) - 1
metrics = compute_metrics(
    result.net_returns, result.benchmark_returns, result.turnover, result.ic,
    cost_drag=gross_ann - net_ann,
)

# IS/OOS split point
oos_frac_used = st.session_state.get("oos_frac", 0.0)
split_date = None
if oos_frac_used > 0:
    ridx = result.net_returns.index
    split_date = ridx[int(len(ridx) * (1 - oos_frac_used))]

tab_ov, tab_win, tab_sec, tab_fac, tab_sig, tab_num, tab_cmp = st.tabs(
    ["Overview", "Winners & Losers", "Sector Lens", "Factor Overlays",
     "Signal Diagnostics", "All the Numbers", "Compare"])

# ---------------- Overview ----------------
with tab_ov:
    m = metrics
    row1 = st.columns(6)
    row1[0].metric("Sharpe", f"{m['sharpe']:.2f}")
    row1[1].metric("Lo-corrected", f"{m['sharpe_lo_corrected']:.2f}",
                   f"p={m['sharpe_p_value']:.3f}", delta_color="off")
    row1[2].metric("Ann. return", f"{m['ann_return']:.1%}")
    row1[3].metric("Ann. vol", f"{m['ann_vol']:.1%}")
    row1[4].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
    row1[5].metric("Hit rate", f"{m['hit_rate']:.1%}")
    row2 = st.columns(6)
    row2[0].metric("Sortino", f"{m['sortino']:.2f}" if m.get("sortino") else "—")
    row2[1].metric("Calmar", f"{m['calmar']:.2f}" if m.get("calmar") else "—")
    row2[2].metric(f"Beta ({bench_label})", f"{m.get('beta', float('nan')):.2f}"
                   if "beta" in m else "—")
    row2[3].metric("CAPM α (ann)", f"{m.get('capm_alpha_ann', 0):.1%}"
                   if "capm_alpha_ann" in m else "—")
    row2[4].metric("Turnover (ann)", f"{m.get('ann_turnover', 0):.0%}"
                   if "ann_turnover" in m else "—")
    row2[5].metric("Cost drag (ann)", f"{m.get('ann_cost_drag', 0):.2%}")

    sig = "✅ significant at 5% (Lo-corrected)" if m["sharpe_p_value"] < 0.05 else \
        "⚠️ NOT significant at 5% (Lo-corrected)"
    lo_ci = m["sharpe_ci_95"]
    st.caption(f"{sig} · bootstrap 95% CI on Sharpe: [{lo_ci[0]:.2f}, {lo_ci[1]:.2f}]"
               + (" — excludes zero ✅" if lo_ci[0] > 0 else " — includes zero ⚠️"))

    if split_date is not None:
        m_is = compute_metrics(result.net_returns.loc[:split_date],
                               result.benchmark_returns)
        m_oos = compute_metrics(result.net_returns.loc[split_date:],
                                result.benchmark_returns)
        if "error" not in m_is and "error" not in m_oos:
            keys = [("sharpe", "Sharpe", "{:.2f}"),
                    ("ann_return", "Ann. return", "{:.1%}"),
                    ("ann_vol", "Ann. vol", "{:.1%}"),
                    ("max_drawdown", "Max DD", "{:.1%}"),
                    ("hit_rate", "Hit rate", "{:.1%}")]
            split_df = pd.DataFrame(
                {"In-sample": [fmt.format(m_is[k]) for k, _, fmt in keys],
                 "Out-of-sample": [fmt.format(m_oos[k]) for k, _, fmt in keys]},
                index=[label for _, label, _ in keys])
            sc1, sc2 = st.columns([1, 2])
            sc1.dataframe(split_df, use_container_width=True)
            ratio = (m_oos["sharpe"] / m_is["sharpe"]
                     if m_is["sharpe"] not in (0, None) else float("nan"))
            verdict_txt = ("holds up out of sample" if ratio > 0.5
                           else "degrades out of sample — likely overfit"
                           if np.isfinite(ratio) else "n/a")
            sc2.markdown(
                f"**OOS/IS Sharpe ratio: {ratio:.2f}** — {verdict_txt}.\n\n"
                f"Split at **{split_date.date()}** (last {oos_frac_used:.0%} held out). "
                "The signal expression never saw these dates when you were "
                "iterating on it — treat the OOS column as the honest number.")

    log_scale = st.toggle("Log scale", value=False)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.74, 0.26], vertical_spacing=0.03)
    gross_eq = (1 + result.gross_returns.fillna(0)).cumprod()
    fig.add_trace(go.Scatter(x=gross_eq.index, y=gross_eq, name="Gross (pre-cost)",
                             line=dict(width=1, dash="dot", color=GRAY),
                             hovertemplate="%{y:.2f}<extra>gross</extra>"),
                  row=1, col=1)
    if result.benchmark_returns is not None:
        beq = (1 + result.benchmark_returns.fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=beq.index, y=beq, name=bench_label,
                                 line=dict(width=1.4, color="#6B7280"),
                                 hovertemplate="%{y:.2f}<extra>" + bench_label + "</extra>"),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity,
                             name="Strategy (net)",
                             line=dict(width=2.4, color=ACCENT),
                             hovertemplate="%{y:.2f}<extra>net</extra>"),
                  row=1, col=1)
    dd = drawdown_series(result.net_returns)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown", fill="tozeroy",
                             line=dict(color=RED, width=1),
                             fillcolor="rgba(240,84,79,0.22)",
                             hovertemplate="%{y:.1%}<extra>drawdown</extra>",
                             showlegend=False),
                  row=2, col=1)
    if split_date is not None:
        fig.add_vrect(x0=split_date, x1=result.equity.index[-1],
                      fillcolor="rgba(91,141,239,0.06)", line_width=0,
                      annotation_text="out-of-sample",
                      annotation_position="top left",
                      annotation_font=dict(size=11, color="#8B93A7"),
                      row=1, col=1)
    style_fig(fig, height=560, title=f"Growth of $1 — net vs gross vs {bench_label}")
    fig.update_yaxes(type="log" if log_scale else "linear", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[dict(count=6, label="6m", step="month", stepmode="backward"),
                     dict(count=1, label="1y", step="year", stepmode="backward"),
                     dict(count=3, label="3y", step="year", stepmode="backward"),
                     dict(step="all", label="All")],
            bgcolor="#151B2B", activecolor="#5B8DEF", font=dict(size=11),
        ),
        row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Winners & Losers (period attribution) ----------------
with tab_win:
    idx = result.net_returns.index
    presets = {"Full backtest": idx[0], "Last 3 years": idx[-1] - pd.DateOffset(years=3),
               "Last year": idx[-1] - pd.DateOffset(years=1),
               "Last 6 months": idx[-1] - pd.DateOffset(months=6),
               "Last 3 months": idx[-1] - pd.DateOffset(months=3),
               "YTD": pd.Timestamp(idx[-1].year, 1, 1)}
    c1, c2 = st.columns([1, 2])
    choice = c1.selectbox("Period", list(presets) + ["Custom"])
    if choice == "Custom":
        p_start, p_end = c2.slider(
            "Custom range", min_value=idx[0].to_pydatetime(),
            max_value=idx[-1].to_pydatetime(),
            value=(idx[0].to_pydatetime(), idx[-1].to_pydatetime()), format="YYYY-MM")
        p_start, p_end = pd.Timestamp(p_start), pd.Timestamp(p_end)
    else:
        p_start, p_end = max(presets[choice], idx[0]), idx[-1]

    period_ret = (1 + result.net_returns.loc[p_start:p_end].fillna(0)).prod() - 1
    bench_ret = None
    if result.benchmark_returns is not None:
        bench_ret = (1 + result.benchmark_returns.loc[p_start:p_end].fillna(0)).prod() - 1
    m1, m2, m3 = st.columns(3)
    m1.metric("Strategy return", f"{period_ret:.1%}")
    m2.metric(f"{bench_label} return", f"{bench_ret:.1%}" if bench_ret is not None else "—")
    m3.metric("Active", f"{period_ret - bench_ret:+.1%}" if bench_ret is not None else "—")

    contrib = position_contribution(result.weights, prices, p_start, p_end)
    if not contrib.empty:
        n_show = st.slider("Names to show per side", 5, 25, 12)
        top = contrib.head(n_show)
        bot = contrib.tail(n_show).iloc[::-1]
        cc1, cc2 = st.columns(2)
        for col, dfc, title in [
            (cc1, top, "Top P&L contributors"),
            (cc2, bot, "Worst P&L contributors"),
        ]:
            fig_c = go.Figure(go.Bar(
                x=dfc["contribution"], y=dfc.index, orientation="h",
                marker=dict(color=diverging_colors(dfc["contribution"]),
                            line=dict(width=0)),
                text=[f"{v:+.2%}" for v in dfc["contribution"]],
                textposition="outside", textfont=dict(size=11),
                cliponaxis=False,
                customdata=np.stack([dfc["own_return"], dfc["avg_weight"],
                                     dfc["days_held"]], axis=1),
                hovertemplate="<b>%{y}</b>: %{x:.2%} contribution<br>"
                              "own return while held: %{customdata[0]:.1%}<br>"
                              "avg weight: %{customdata[1]:.2%} · "
                              "days held: %{customdata[2]}<extra></extra>"))
            style_fig(fig_c, height=32 * len(dfc) + 90, title=title,
                      hover="closest", show_legend=False)
            fig_c.update_layout(xaxis_tickformat=".1%", bargap=0.32,
                                yaxis=dict(autorange="reversed"))
            fig_c.update_xaxes(range=[dfc["contribution"].min() * 1.35 if
                                      dfc["contribution"].min() < 0 else 0,
                                      max(dfc["contribution"].max(), 0) * 1.35])
            col.plotly_chart(fig_c, use_container_width=True)
        st.caption("Contribution = Σ(weight × daily return) while held. Hover for the "
                   "stock's own return, average weight, and days held.")

    st.subheader("Universe: best & worst performers (held or not)")
    uni = universe_performance(prices, p_start, p_end)
    held_now = set(contrib.index) if not contrib.empty else set()
    uni_df = pd.DataFrame({"return": uni})
    uni_df["traded by strategy"] = ["✅" if t in held_now else "" for t in uni_df.index]
    u1, u2 = st.columns(2)
    u1.dataframe(uni_df.head(15).style.format({"return": "{:.1%}"}),
                 use_container_width=True)
    u2.dataframe(uni_df.tail(15).iloc[::-1].style.format({"return": "{:.1%}"}),
                 use_container_width=True)
    missed = uni_df.head(15)["traded by strategy"].eq("").sum()
    st.caption(f"{missed} of the period's top 15 performers were never held — "
               "if that number is high, the signal is missing the winners, "
               "not just riding the market.")

    if not factors.empty:
        st.subheader("Where the return came from (factor decomposition)")
        fc = factor_contribution(result.net_returns, factors, p_start, p_end)
        if not fc.empty:
            plot_fc = fc.drop(index="Total").sort_values(
                "contribution", key=lambda s: s.abs(), ascending=False)
            fig_fc = go.Figure(go.Waterfall(
                x=list(plot_fc.index) + ["Total"],
                y=list(plot_fc["contribution"]) + [0],
                measure=["relative"] * len(plot_fc) + ["total"],
                text=[f"{v:+.1%}" for v in plot_fc["contribution"]]
                     + [f"{period_ret:.1%}"],
                textposition="outside", textfont=dict(size=11), cliponaxis=False,
                connector=dict(line=dict(color="#2A3247", width=1)),
                increasing=dict(marker=dict(color="rgba(46,194,126,0.85)")),
                decreasing=dict(marker=dict(color="rgba(240,84,79,0.85)")),
                totals=dict(marker=dict(color=ACCENT)),
            ))
            style_fig(fig_fc, height=380, hover="closest", show_legend=False,
                      title=f"Period return {period_ret:.1%} decomposed "
                            "(period betas × factor returns)")
            fig_fc.update_layout(yaxis_tickformat=".0%", waterfallgap=0.35)
            st.plotly_chart(fig_fc, use_container_width=True)
            st.caption("Big 'Residual (alpha)' bar = the strategy did something factors "
                       "don't explain in this period. Big factor bars = it was riding "
                       "(or fighting) that factor.")

    st.subheader("Best & worst stretches")
    bw = best_worst_windows(result.net_returns.loc[p_start:p_end])
    if not bw.empty:
        st.dataframe(
            bw.style.format({"return": "{:.1%}"})
            .map(lambda v: "color: seagreen" if v == "best"
                 else ("color: crimson" if v == "worst" else ""), subset=["type"]),
            use_container_width=True, hide_index=True)

    st.subheader("Stock drilldown")
    dd_options = (list(contrib.index) if not contrib.empty
                  else list(prices.columns))
    sel_t = st.selectbox("Ticker — price, holding periods, and the signal that "
                         "drove them", dd_options)
    if sel_t:
        px_s = prices[sel_t].loc[p_start:p_end]
        w_s = result.weights[sel_t].reindex(px_s.index).fillna(0)
        sig_rank = st.session_state["signal"].rank(axis=1, pct=True)[sel_t] \
            .reindex(px_s.index)
        fig_dd2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.65, 0.35], vertical_spacing=0.04,
                                specs=[[{}], [{"secondary_y": True}]])
        fig_dd2.add_trace(go.Scatter(x=px_s.index, y=px_s, name="Price",
                                     line=dict(color=GRAY, width=1.4),
                                     hovertemplate="%{y:.2f}<extra>price</extra>"),
                          row=1, col=1)
        long_px = px_s.where(w_s > 1e-9)
        short_px = px_s.where(w_s < -1e-9)
        fig_dd2.add_trace(go.Scatter(x=long_px.index, y=long_px, name="Held long",
                                     line=dict(color=GREEN, width=2.4),
                                     hovertemplate="%{y:.2f}<extra>long</extra>"),
                          row=1, col=1)
        fig_dd2.add_trace(go.Scatter(x=short_px.index, y=short_px, name="Held short",
                                     line=dict(color=RED, width=2.4),
                                     hovertemplate="%{y:.2f}<extra>short</extra>"),
                          row=1, col=1)
        fig_dd2.add_trace(go.Scatter(x=sig_rank.index, y=sig_rank,
                                     name="Signal rank (0–1)",
                                     line=dict(color=CYAN, width=1.6),
                                     hovertemplate="%{y:.2f}<extra>signal rank</extra>"),
                          row=2, col=1)
        fig_dd2.add_trace(go.Scatter(x=w_s.index, y=w_s, name="Weight",
                                     fill="tozeroy",
                                     line=dict(color=ACCENT, width=1),
                                     fillcolor="rgba(91,141,239,0.18)",
                                     hovertemplate="%{y:.2%}<extra>weight</extra>"),
                          row=2, col=1, secondary_y=True)
        style_fig(fig_dd2, height=480, title=f"{sel_t} — when it was held and why")
        fig_dd2.update_yaxes(range=[0, 1], row=2, col=1, secondary_y=False)
        fig_dd2.update_yaxes(tickformat=".1%", showgrid=False, row=2, col=1,
                             secondary_y=True)
        st.plotly_chart(fig_dd2, use_container_width=True)
        st.caption("Top: price, colored green/red while the strategy held it. "
                   "Bottom: the stock's cross-sectional signal rank (cyan, 0–1) "
                   "and its portfolio weight (blue area). If the green stretches sit "
                   "on downtrends, the signal is entering too early or exiting too late.")

# ---------------- Sector Lens ----------------
with tab_sec:
    sub_of = {t: SUBSECTOR.get(t, "Other") for t in result.weights.columns}
    mapped = [t for t in result.weights.columns if t in SUBSECTOR]
    if not mapped:
        st.info("No insurance names in this universe — pick an Insurance "
                "universe in the sidebar to use the Sector Lens.")
    else:
        w = result.weights
        by_sub = w.T.groupby(pd.Series(sub_of)).sum().T
        by_sub = by_sub.loc[:, by_sub.abs().max() > 1e-9]
        gross_sub = w.abs().T.groupby(pd.Series(sub_of)).sum().T
        gross_sub = gross_sub.loc[:, gross_sub.max() > 1e-9]

        # Gross exposure mix over time (stacked, weekly)
        gw = gross_sub.resample("W").mean()
        share = gw.div(gw.sum(axis=1).replace(0, np.nan), axis=0)
        fig_mix = go.Figure()
        for subname in share.columns:
            fig_mix.add_trace(go.Scatter(
                x=share.index, y=share[subname], name=subname,
                stackgroup="mix", mode="none",
                fillcolor=with_alpha(SUBSECTOR_COLORS.get(subname, GRAY), 0.65),
                hovertemplate="%{y:.0%}<extra>" + subname + "</extra>"))
        style_fig(fig_mix, height=360, ytickformat=".0%",
                  title="Gross exposure mix by subsector (weekly avg)")
        fig_mix.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_mix, use_container_width=True)

        sc1, sc2 = st.columns(2)
        # Current net weight by subsector
        cur_net = by_sub.iloc[-1].sort_values()
        fig_net = go.Figure(go.Bar(
            x=cur_net.values, y=cur_net.index, orientation="h",
            marker=dict(color=diverging_colors(cur_net.values),
                        line=dict(width=0)),
            text=[f"{v:+.1%}" for v in cur_net.values], textposition="outside",
            cliponaxis=False))
        style_fig(fig_net, height=320, hover="closest", show_legend=False,
                  title="Current net weight by subsector")
        fig_net.update_layout(xaxis_tickformat=".0%", bargap=0.35)
        sc1.plotly_chart(fig_net, use_container_width=True)

        # P&L contribution by subsector (full backtest)
        contrib_full = position_contribution(result.weights, prices)
        if not contrib_full.empty:
            csub = contrib_full["contribution"].groupby(
                contrib_full.index.map(lambda t: sub_of.get(t, "Other"))).sum() \
                .sort_values()
            fig_cs = go.Figure(go.Bar(
                x=csub.values, y=csub.index, orientation="h",
                marker=dict(color=diverging_colors(csub.values),
                            line=dict(width=0)),
                text=[f"{v:+.2%}" for v in csub.values], textposition="outside",
                cliponaxis=False))
            style_fig(fig_cs, height=320, hover="closest", show_legend=False,
                      title="P&L contribution by subsector (full backtest)")
            fig_cs.update_layout(xaxis_tickformat=".1%", bargap=0.35)
            sc2.plotly_chart(fig_cs, use_container_width=True)

        # Subsector performance: equal-weight cumulative return per subsector
        st.subheader("Subsector tape (equal-weight, growth of $1)")
        rets_all = prices[mapped].pct_change(fill_method=None)
        fig_tape = go.Figure()
        for subname in sorted({sub_of[t] for t in mapped}):
            members = [t for t in mapped if sub_of[t] == subname]
            if len(members) < 2:
                continue
            eq = (1 + rets_all[members].mean(axis=1).fillna(0)).cumprod()
            fig_tape.add_trace(go.Scatter(
                x=eq.index, y=eq, name=f"{subname} ({len(members)})",
                line=dict(width=1.8, color=SUBSECTOR_COLORS.get(subname, GRAY)),
                hovertemplate="%{y:.2f}<extra>" + subname + "</extra>"))
        style_fig(fig_tape, height=400)
        st.plotly_chart(fig_tape, use_container_width=True)
        st.caption("Where the cycle is: brokers compounding through everything, "
                   "P&C riding the hard market, life tracking rates — check the "
                   "strategy's subsector mix against which tapes are working.")

# ---------------- Factor overlays ----------------
with tab_fac:
    if factors.empty:
        st.warning("No factor data — seed factors in Data Explorer.")
    else:
        # Factor list is discovered from the data source: FF5+MOM locally,
        # Axioma WW4 style factors on the desk, and anything a custom factor
        # loader writes shows up here automatically.
        avail = [c for c in factors.columns if str(c).upper() != "RF"]
        chosen = st.multiselect("Factors", avail, default=avail)
        fac = factors[chosen + (["RF"] if "RF" in factors.columns else [])]

        reg = factor_regression(result.net_returns, fac)
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
            st.caption("Newey-West (HAC) standard errors. α is what's left after the "
                       "factors — if it's not significant, returns are factor tilts.")

            beta_df = pd.DataFrame(reg["betas"]).T
            beta_df.columns = ["beta", "t-stat", "p-value"]
            st.dataframe(
                beta_df.style.format({"beta": "{:.3f}", "t-stat": "{:.2f}",
                                      "p-value": "{:.3f}"})
                .map(lambda v: "background-color: rgba(255,165,0,.25)"
                     if isinstance(v, float) and abs(v) > 2 else "",
                     subset=["t-stat"]),
                use_container_width=True,
            )

            # --- Overlay: actual equity vs what the factors alone would produce ---
            betas = {k: v["beta"] for k, v in reg["betas"].items()}
            Xa = fac.reindex(result.net_returns.index).dropna()
            if not Xa.empty:
                rf_s = Xa["RF"] if "RF" in Xa.columns else 0.0
                implied = sum(Xa[k] * b for k, b in betas.items()) + rf_s
                strat = result.net_returns.reindex(Xa.index).fillna(0)
                resid = strat - implied
                fig_ov = go.Figure()
                fig_ov.add_trace(go.Scatter(
                    x=Xa.index, y=(1 + implied).cumprod(),
                    name="Factor replication (β × factors)",
                    line=dict(width=1.6, color=GOLD, dash="dash"),
                    hovertemplate="%{y:.2f}<extra>factor-implied</extra>"))
                fig_ov.add_trace(go.Scatter(
                    x=Xa.index, y=(1 + strat).cumprod(), name="Strategy (net)",
                    line=dict(width=2.4, color=ACCENT),
                    hovertemplate="%{y:.2f}<extra>strategy</extra>"))
                fig_ov.add_trace(go.Scatter(
                    x=Xa.index, y=(1 + resid).cumprod(),
                    name="Residual α (strategy − replication)",
                    line=dict(width=1.6, color=GREEN),
                    hovertemplate="%{y:.2f}<extra>residual α</extra>"))
                fig_ov.add_hline(y=1, line_dash="dot",
                                 line_color="rgba(255,255,255,0.25)")
                style_fig(fig_ov, height=420,
                          title="Factor overlay — can a static factor portfolio "
                                "replicate this strategy?")
                st.plotly_chart(fig_ov, use_container_width=True)
                st.caption("Blue = your strategy. Gold = a static portfolio of the "
                           "factors with the same betas. Green = what's left after "
                           "subtracting it. A flat/falling green line means the "
                           "strategy is replicable with factor ETFs; a rising green "
                           "line is genuine alpha.")

            roll = rolling_factor_betas(result.net_returns, fac, window=126)
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
                st.plotly_chart(fig_b, use_container_width=True)

                alpha_roll = roll["alpha_ann"]
                fig_a = go.Figure(go.Scatter(
                    x=alpha_roll.index, y=alpha_roll, fill="tozeroy",
                    line=dict(color=GREEN, width=1.8),
                    fillcolor="rgba(46,194,126,0.15)",
                    hovertemplate="%{y:.1%}<extra>rolling α</extra>"))
                style_fig(fig_a, height=260, ytickformat=".0%",
                          title="Rolling annualized factor α", show_legend=False)
                st.plotly_chart(fig_a, use_container_width=True)

            hm = factor_exposure_heatmap_data(result.weights, prices, fac)
            if not hm.empty:
                fig_h = go.Figure(go.Heatmap(
                    z=hm.values, x=list(hm.columns), y=list(hm.index),
                    colorscale=[[0, "rgba(240,84,79,0.9)"],
                                [0.5, "rgba(21,27,43,1)"],
                                [1, "rgba(46,194,126,0.9)"]],
                    zmid=0,
                    text=np.round(hm.values, 3), texttemplate="%{text}",
                    textfont=dict(size=10),
                    colorbar=dict(thickness=10, outlinewidth=0),
                    hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>"))
                style_fig(fig_h, height=max(320, 22 * len(hm)), hover="closest",
                          show_legend=False,
                          title="Current holdings — weight × factor beta "
                                "(exposure contribution)")
                st.plotly_chart(fig_h, use_container_width=True)

# ---------------- Signal diagnostics ----------------
with tab_sig:
    icd = result.ic.dropna()
    if len(icd) > 5:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean rank IC", f"{icd.mean():.4f}")
        c2.metric("IC t-stat", f"{icd.mean() / icd.std() * np.sqrt(len(icd)):.2f}")
        c3.metric("IC hit rate", f"{(icd > 0).mean():.0%}")

        fig_ic = go.Figure()
        fig_ic.add_trace(go.Bar(x=icd.index, y=icd, name="Rank IC",
                                marker=dict(color=diverging_colors(icd),
                                            line=dict(width=0))))
        fig_ic.add_trace(go.Scatter(x=icd.index, y=icd.cumsum(), name="Cumulative IC",
                                    yaxis="y2", line=dict(color=GOLD, width=2.2)))
        style_fig(fig_ic, height=360,
                  title="Rank IC per rebalance (signal vs forward return)")
        fig_ic.update_layout(yaxis2=dict(overlaying="y", side="right",
                                         showgrid=False, zeroline=False))
        st.plotly_chart(fig_ic, use_container_width=True)
        st.caption("A monotonically rising cumulative IC = stable predictive power. "
                   "Flat or regime-y = signal works only sometimes.")

    if not result.quantile_returns.empty:
        qr = result.quantile_returns
        fig_q = go.Figure(go.Bar(
            x=qr.index, y=qr.values,
            marker=dict(color=diverging_colors(qr.values), line=dict(width=0)),
            text=[f"{v:+.1%}" for v in qr.values], textposition="outside",
            textfont=dict(size=11), cliponaxis=False))
        style_fig(fig_q, height=320, ytickformat=".0%", hover="closest",
                  show_legend=False,
                  title="Annualized forward return by signal quintile "
                        "(Q5 = highest signal)")
        fig_q.update_layout(bargap=0.4)
        st.plotly_chart(fig_q, use_container_width=True)
        st.caption("You want monotonic bars. If Q5 ≈ Q1 the signal has no spread; "
                   "if only the tails work, trade narrower quantiles.")

    to = result.turnover
    if len(to):
        fig_t = go.Figure(go.Scatter(
            x=to.index, y=to.rolling(10).mean(), fill="tozeroy",
            line=dict(color=PURPLE, width=1.8),
            fillcolor="rgba(155,126,222,0.15)",
            hovertemplate="%{y:.1%}<extra>turnover</extra>"))
        style_fig(fig_t, height=240, ytickformat=".0%", show_legend=False,
                  title="One-way turnover per rebalance (10-rebalance MA)")
        st.plotly_chart(fig_t, use_container_width=True)

# ---------------- All the numbers ----------------
with tab_num:
    st.subheader("Monthly returns")
    mt = monthly_return_table(result.net_returns)
    st.dataframe(
        mt.style.format("{:.1%}", na_rep="")
        .background_gradient(cmap="RdYlGn", vmin=-0.08, vmax=0.08, axis=None),
        use_container_width=True,
    )
    st.subheader("Full metric dump")
    flat = {k: (f"{v:.4f}" if isinstance(v, float) else str(v))
            for k, v in metrics.items() if not isinstance(v, tuple)}
    ci = metrics.get("sharpe_ci_95")
    if ci:
        flat["sharpe_ci_95"] = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
    st.dataframe(pd.Series(flat, name="value").to_frame(), use_container_width=True)

    st.subheader("Latest positions")
    lw = result.weights.iloc[-1]
    lw = lw[lw.abs() > 1e-6].sort_values(ascending=False)
    st.dataframe(lw.rename("weight").to_frame().style.format("{:.2%}"),
                 use_container_width=True, height=300)

    st.subheader("Export")
    import json as _json
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("⬇ Daily returns (CSV)",
                       result.net_returns.rename("net_return").to_csv().encode(),
                       "strategy_returns.csv", "text/csv", key="dl_ret")
    d2.download_button("⬇ Daily weights (CSV)",
                       result.weights.to_csv().encode(),
                       "strategy_weights.csv", "text/csv", key="dl_w")
    d3.download_button("⬇ Metrics (JSON)",
                       _json.dumps({**{k: v for k, v in metrics.items()
                                       if not isinstance(v, tuple)},
                                    "expression": st.session_state["expression"],
                                    "config": cfg.to_dict()},
                                   indent=2, default=str).encode(),
                       "metrics.json", "application/json", key="dl_m")
    d4.download_button("⬇ Rank IC series (CSV)",
                       result.ic.rename("rank_ic").to_csv().encode(),
                       "rank_ic.csv", "text/csv", key="dl_ic")

# ---------------- Compare ----------------
with tab_cmp:
    st.markdown("Snapshot runs here, then re-run with a different signal or "
                "config to compare them head-to-head.")
    cc1, cc2 = st.columns([3, 1])
    snap_name = cc1.text_input("Label for current run",
                               value=st.session_state["expression"][:48])
    if cc2.button("➕ Snapshot current run", use_container_width=True):
        st.session_state.setdefault("compare", {})[snap_name] = {
            "returns": result.net_returns,
            "metrics": {k: v for k, v in metrics.items()
                        if not isinstance(v, (tuple, str))},
            "expression": st.session_state["expression"],
            "config": cfg.to_dict(),
        }
        st.rerun()

    runs = st.session_state.get("compare", {})
    if not runs:
        st.info("No snapshots yet.")
    else:
        palette = [ACCENT, GOLD, GREEN, PURPLE, CYAN, RED, GRAY]
        fig_cmp = go.Figure()
        for i, (name, r) in enumerate(runs.items()):
            eq = (1 + r["returns"].fillna(0)).cumprod()
            fig_cmp.add_trace(go.Scatter(
                x=eq.index, y=eq, name=name[:40],
                line=dict(width=2, color=palette[i % len(palette)]),
                hovertemplate="%{y:.2f}<extra>" + name[:24] + "</extra>"))
        if result.benchmark_returns is not None:
            beq = (1 + result.benchmark_returns.fillna(0)).cumprod()
            fig_cmp.add_trace(go.Scatter(x=beq.index, y=beq, name=bench_label,
                                         line=dict(width=1.2, color="#6B7280",
                                                   dash="dot")))
        style_fig(fig_cmp, height=440, title="Snapshotted runs — growth of $1")
        st.plotly_chart(fig_cmp, use_container_width=True)

        cmp_keys = [("sharpe", "{:.2f}"), ("sharpe_lo_corrected", "{:.2f}"),
                    ("sharpe_p_value", "{:.3f}"), ("ann_return", "{:.1%}"),
                    ("ann_vol", "{:.1%}"), ("max_drawdown", "{:.1%}"),
                    ("beta", "{:.2f}"), ("capm_alpha_ann", "{:.1%}"),
                    ("ann_turnover", "{:.0%}"), ("ann_cost_drag", "{:.2%}"),
                    ("ic_tstat", "{:.2f}"), ("hit_rate", "{:.1%}")]
        cmp_df = pd.DataFrame({
            name: {k: (fmt.format(r["metrics"][k])
                       if r["metrics"].get(k) is not None else "—")
                   for k, fmt in cmp_keys}
            for name, r in runs.items()})
        st.dataframe(cmp_df, use_container_width=True)

        with st.expander("Expressions & configs"):
            for name, r in runs.items():
                st.markdown(f"**{name}**")
                st.code(r["expression"], language="python")
        rm1, rm2 = st.columns(2)
        drop = rm1.selectbox("Remove a snapshot", ["—"] + list(runs))
        if drop != "—":
            del st.session_state["compare"][drop]
            st.rerun()
        if rm2.button("Clear all snapshots"):
            st.session_state["compare"] = {}
            st.rerun()

# ---------------- Save as theory ----------------
st.divider()
with st.expander("💾 Save this run to the Theory Journal"):
    tname = st.text_input("Theory name", value=st.session_state.get("expression", "")[:60])
    thesis = st.text_area("Hypothesis — what do you believe and why?",
                          placeholder="e.g. Stocks near 52-week highs keep outperforming "
                                      "because anchoring delays repricing.")
    verdict_auto = ("supported" if metrics["sharpe_p_value"] < 0.05
                    and metrics["sharpe"] > 0 else "not supported")
    verdict = st.selectbox("Verdict", ["supported", "not supported", "inconclusive"],
                           index=["supported", "not supported"].index(verdict_auto)
                           if verdict_auto in ["supported", "not supported"] else 2)
    if st.button("Save theory"):
        keep_keys = ["sharpe", "sharpe_lo_corrected", "sharpe_p_value", "ann_return",
                     "ann_vol", "max_drawdown", "capm_alpha_ann", "beta",
                     "ann_turnover", "ann_cost_drag", "ic_mean", "ic_tstat",
                     "hit_rate", "start", "end", "n_days"]
        _theories().save_theory(
            name=tname, hypothesis=thesis,
            expression=st.session_state["expression"],
            config=cfg.to_dict(),
            metrics={k: metrics[k] for k in keep_keys if k in metrics},
            verdict=verdict,
        )
        st.success("Saved — see the Theory Journal page.")
