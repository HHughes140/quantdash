"""Period attribution: what did well, what did poorly, and why.

Three lenses over any sub-period of a backtest:
  1. Position contribution — which holdings made/lost the money (weight x return).
  2. Universe performance — best/worst names overall, held or not.
  3. Factor contribution — how much of the period return came from each factor
     exposure vs residual alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def position_contribution(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Per-ticker contribution to strategy return over [start, end].

    Returns a DataFrame indexed by ticker: contribution (sum of w*ret),
    avg_weight, days_held, own_return (the stock's own total return while held).
    """
    rets = prices.pct_change(fill_method=None)
    w = weights.reindex(columns=prices.columns).fillna(0.0)
    sl = slice(start, end)
    w, rets = w.loc[sl], rets.loc[sl]

    contrib = (w * rets).sum()
    held_mask = w.abs() > 1e-9
    days_held = held_mask.sum()
    avg_w = w[held_mask].mean()

    own_ret = {}
    for t in prices.columns:
        mask = held_mask[t]
        if mask.any():
            r = rets[t][mask]
            own_ret[t] = float((1 + r.fillna(0)).prod() - 1)
    own = pd.Series(own_ret)

    out = pd.DataFrame({
        "contribution": contrib,
        "avg_weight": avg_w,
        "days_held": days_held,
        "own_return": own,
    })
    out = out[out["days_held"] > 0].sort_values("contribution", ascending=False)
    return out


def universe_performance(prices: pd.DataFrame, start=None, end=None) -> pd.Series:
    """Total return of every name in the universe over [start, end]."""
    p = prices.loc[slice(start, end)].dropna(axis=1, how="all")
    first = p.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    last = p.apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    return (last / first - 1).dropna().sort_values(ascending=False)


def factor_contribution(
    returns: pd.Series,
    factors: pd.DataFrame,
    start=None,
    end=None,
    rf_col: str = "RF",
) -> pd.DataFrame:
    """Decompose the period's total return into factor pieces + residual.

    Betas are estimated over the chosen period itself; each factor's
    contribution is beta_k * sum(factor_k returns). Returns a DataFrame with
    rows per factor plus 'Residual (alpha)' and 'Total', column 'contribution'.
    """
    sl = slice(start, end)
    r = returns.loc[sl].dropna()
    f = factors.reindex(r.index)
    if rf_col in f.columns:
        rf = f[rf_col].fillna(0)
        y = r - rf
        X = f.drop(columns=[rf_col])
    else:
        rf = pd.Series(0.0, index=r.index)
        y, X = r, f
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(data) < 40:
        return pd.DataFrame()

    Xm = np.column_stack([np.ones(len(data)), data.drop(columns="y").values])
    coef, *_ = np.linalg.lstsq(Xm, data["y"].values, rcond=None)
    names = list(data.drop(columns="y").columns)

    rows = {}
    for k, name in enumerate(names):
        rows[name] = float(coef[k + 1] * data[name].sum())
    total = float(r.sum())
    explained = sum(rows.values()) + float(rf.reindex(data.index).sum())
    rows["Risk-free"] = float(rf.reindex(data.index).sum())
    rows["Residual (alpha)"] = total - explained
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["contribution"])
    out["beta"] = [float(coef[k + 1]) for k in range(len(names))] + [np.nan, np.nan]
    out.loc["Total"] = [total, np.nan]
    return out


def best_worst_windows(
    returns: pd.Series, window: int = 21, n: int = 5
) -> pd.DataFrame:
    """Non-overlapping best and worst rolling windows (default ~1 month)."""
    roll = (1 + returns.fillna(0)).rolling(window).apply(np.prod, raw=True) - 1
    roll = roll.dropna()
    picked, rows = [], []
    for label, series in [("best", roll.sort_values(ascending=False)),
                          ("worst", roll.sort_values())]:
        count = 0
        for end_dt, val in series.items():
            start_dt = returns.index[max(0, returns.index.get_loc(end_dt) - window + 1)]
            if any(not (end_dt < s or start_dt > e) for s, e in picked):
                continue
            picked.append((start_dt, end_dt))
            rows.append({"type": label, "start": start_dt.date(), "end": end_dt.date(),
                         "return": float(val)})
            count += 1
            if count >= n:
                break
    return pd.DataFrame(rows).sort_values("return", ascending=False)
