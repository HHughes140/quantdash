"""Liquidity and capacity analysis.

Answers "at what AUM does this strategy stop working?" using a square-root
market-impact model: per-name impact (in return terms) is

    impact = |trade as % of NAV| x daily_vol x sqrt(participation)

where participation = traded dollars / average daily dollar volume. The base
`cost_bps` spread cost is already inside the backtest's net returns; this adds
the size-dependent piece on top.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

ANN = 252


def dollar_adv(prices: pd.DataFrame, volume: pd.DataFrame,
               window: int = 20, volume_is_dollars: bool = False) -> pd.DataFrame:
    """Average daily dollar volume panel. On the Axioma feed the volume field
    is already a dollar ADV; from yfinance it's shares, so multiply by price."""
    if volume_is_dollars:
        return volume.rolling(window).mean()
    return (volume * prices).rolling(window).mean()


def capacity_analysis(
    net_returns: pd.Series,
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    adv_dollars: pd.DataFrame,
    aum_grid: Sequence[float] = (10e6, 50e6, 100e6, 250e6, 500e6, 1e9),
) -> pd.DataFrame:
    """Per-AUM impact drag and impact-adjusted Sharpe.

    Returns a DataFrame indexed by AUM with: ann_impact_drag, sharpe_net (as
    backtested, size-free), sharpe_after_impact, median/p95 participation.
    """
    dw = weights.diff().abs()
    common = dw.index.intersection(net_returns.index)
    dw = dw.loc[common]
    adv = adv_dollars.reindex(index=common, columns=weights.columns).ffill()
    daily_vol = prices.pct_change(fill_method=None).rolling(21).std() \
        .reindex(index=common, columns=weights.columns)
    base = net_returns.loc[common]
    sd = base.std()
    base_sharpe = float(base.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan

    rows = []
    for aum in aum_grid:
        traded = dw * aum
        part = (traded / adv).replace([np.inf, -np.inf], np.nan)
        part_c = part.clip(upper=1.0)
        impact = (dw * daily_vol * np.sqrt(part_c)).sum(axis=1).fillna(0.0)
        adj = base - impact
        sd_a = adj.std()
        traded_days = part[dw > 0]
        rows.append({
            "aum": aum,
            "ann_impact_drag": float(impact.mean() * ANN),
            "sharpe_after_impact": float(adj.mean() / sd_a * np.sqrt(ANN))
            if sd_a and sd_a > 0 else np.nan,
            "median_participation": float(np.nanmedian(traded_days.values))
            if np.isfinite(traded_days.values).any() else np.nan,
            "p95_participation": float(np.nanpercentile(traded_days.values, 95))
            if np.isfinite(traded_days.values).any() else np.nan,
        })
    out = pd.DataFrame(rows).set_index("aum")
    out["sharpe_net"] = base_sharpe
    return out
