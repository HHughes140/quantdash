"""Vectorized portfolio backtester.

Design goals: no lookahead (weights formed from signal at close of rebalance
day t apply from t+1), explicit transaction costs on traded notional, and
fast enough (<1s for 150 tickers x 10y) for interactive theory testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    mode: str = "long_short"          # long_short | long_only | signal_weight
    quantile: float = 0.2             # top/bottom fraction for quantile modes
    rebalance_every: int = 5          # trading days
    cost_bps: float = 5.0             # one-way cost per unit of traded notional
    max_weight: float = 0.10          # per-name cap (abs)
    min_names: int = 10               # skip dates with fewer valid signals
    vol_target: Optional[float] = None  # e.g. 0.10 -> scale leverage to 10% ann vol
    benchmark: str = "SPY"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame            # daily effective weights
    turnover: pd.Series              # one-way turnover on rebalance days
    ic: pd.Series                    # rank IC of signal vs fwd returns per rebalance
    quantile_returns: pd.Series      # mean fwd return per signal quintile (annualized)
    benchmark_returns: Optional[pd.Series]
    config: BacktestConfig
    equity: pd.Series = field(init=False)

    def __post_init__(self):
        self.equity = (1 + self.net_returns.fillna(0)).cumprod()


def _weights_from_signal(sig_row: pd.Series, cfg: BacktestConfig) -> pd.Series:
    valid = sig_row.dropna()
    if len(valid) < cfg.min_names:
        return pd.Series(dtype=float)

    if cfg.mode == "long_only":
        cutoff = valid.quantile(1 - cfg.quantile)
        chosen = valid[valid >= cutoff]
        w = pd.Series(1.0 / len(chosen), index=chosen.index)
    elif cfg.mode == "long_short":
        top = valid[valid >= valid.quantile(1 - cfg.quantile)]
        bot = valid[valid <= valid.quantile(cfg.quantile)]
        w = pd.concat([
            pd.Series(0.5 / len(top), index=top.index),
            pd.Series(-0.5 / len(bot), index=bot.index),
        ])
        w = w.groupby(w.index).sum()  # a name in both tails nets out
    elif cfg.mode == "signal_weight":
        z = (valid - valid.mean()) / valid.std()
        z = z.clip(-3, 3)
        denom = z.abs().sum()
        if denom == 0:
            return pd.Series(dtype=float)
        w = z / denom
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    w = w.clip(-cfg.max_weight, cfg.max_weight)
    # renormalize gross to 1 after capping
    gross = w.abs().sum()
    if gross > 0:
        w = w / gross
    return w


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    benchmark_prices: Optional[pd.Series] = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    prices = prices.sort_index()
    signal = signal.reindex(index=prices.index, columns=prices.columns)
    rets = prices.pct_change(fill_method=None)

    # Rebalance dates: every Nth trading day from the first date with enough signal
    valid_counts = signal.notna().sum(axis=1)
    valid_dates = valid_counts[valid_counts >= cfg.min_names].index
    if len(valid_dates) < 2:
        raise ValueError(
            "Not enough valid signal history — check lookback windows vs data range."
        )
    all_dates = prices.index
    start_pos = all_dates.get_loc(valid_dates[0])
    rebal_positions = range(start_pos, len(all_dates), cfg.rebalance_every)
    rebal_dates = [all_dates[i] for i in rebal_positions]

    # Target weights at each rebalance close
    target = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)
    for dt in rebal_dates:
        w = _weights_from_signal(signal.loc[dt], cfg)
        if not w.empty:
            target.loc[dt, w.index] = w.values

    # Effective daily weights: hold targets, applied from the next trading day
    daily_w = target.reindex(all_dates).ffill().shift(1).fillna(0.0)

    gross_ret = (daily_w * rets).sum(axis=1)

    # Costs: charged when weights change (the day new targets take effect)
    dw = daily_w.diff().abs().sum(axis=1).fillna(0.0)
    costs = dw * (cfg.cost_bps / 1e4)
    net_ret = gross_ret - costs
    turnover = (dw / 2).loc[dw > 0]

    # Optional vol targeting (63d trailing, capped 3x leverage)
    if cfg.vol_target:
        realized = net_ret.rolling(63).std() * np.sqrt(252)
        lev = (cfg.vol_target / realized).clip(upper=3.0).shift(1).fillna(1.0)
        net_ret = net_ret * lev
        gross_ret = gross_ret * lev

    # Rank IC: signal at rebalance vs forward return to next rebalance
    ic_vals = {}
    for a, b in zip(rebal_dates[:-1], rebal_dates[1:]):
        fwd = prices.loc[b] / prices.loc[a] - 1
        s = signal.loc[a]
        pair = pd.concat([s, fwd], axis=1, keys=["sig", "fwd"]).dropna()
        if len(pair) >= cfg.min_names:
            ic_vals[b] = pair["sig"].corr(pair["fwd"], method="spearman")
    ic = pd.Series(ic_vals, dtype=float)

    # Quintile forward returns (annualized), full sample
    q_rets = _quantile_profile(prices, signal, rebal_dates, n_q=5)

    bench = None
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench = benchmark_prices.reindex(all_dates).pct_change(fill_method=None)
        bench = bench.loc[net_ret.index]

    # Trim to live period
    live = daily_w.abs().sum(axis=1) > 0
    first_live = live.idxmax() if live.any() else all_dates[0]
    sl = slice(first_live, None)

    return BacktestResult(
        net_returns=net_ret.loc[sl],
        gross_returns=gross_ret.loc[sl],
        weights=daily_w.loc[sl],
        turnover=turnover,
        ic=ic,
        quantile_returns=q_rets,
        benchmark_returns=bench.loc[sl] if bench is not None else None,
        config=cfg,
    )


def _quantile_profile(prices, signal, rebal_dates, n_q=5) -> pd.Series:
    """Mean annualized forward return per signal quantile bucket."""
    rows = []
    for a, b in zip(rebal_dates[:-1], rebal_dates[1:]):
        fwd = prices.loc[b] / prices.loc[a] - 1
        s = signal.loc[a]
        pair = pd.concat([s, fwd], axis=1, keys=["sig", "fwd"]).dropna()
        if len(pair) < n_q * 2:
            continue
        try:
            pair["q"] = pd.qcut(pair["sig"].rank(method="first"), n_q,
                                labels=range(1, n_q + 1))
        except ValueError:
            continue
        days = max((b - a).days * (252 / 365.25), 1)
        rows.append(pair.groupby("q", observed=True)["fwd"].mean() * (252 / days))
    if not rows:
        return pd.Series(dtype=float)
    out = pd.concat(rows, axis=1).mean(axis=1)
    out.index = [f"Q{int(i)}" for i in out.index]
    return out
