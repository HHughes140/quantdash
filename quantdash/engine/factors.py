"""Factor overlay analytics: full-sample and rolling factor regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

ANN = 252


def factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    rf_col: str = "RF",
) -> dict:
    """Newey-West OLS of strategy excess returns on factor returns.

    factors: date x factor panel of daily decimal returns (e.g. MKT_RF, SMB,
    HML, RMW, CMA, MOM, RF). Returns dict with alpha (annualized), betas,
    t-stats, p-values, and R^2.
    """
    df = factors.reindex(returns.index).dropna(how="all")
    y = returns.reindex(df.index)
    if rf_col in df.columns:
        y = y - df[rf_col]
        X = df.drop(columns=[rf_col])
    else:
        X = df
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(data) < 60:
        return {"error": "not enough overlapping observations"}
    y_, X_ = data["y"], sm.add_constant(data.drop(columns=["y"]))
    lags = int(np.floor(4 * (len(data) / 100) ** (2 / 9)))  # Newey-West rule of thumb
    model = sm.OLS(y_, X_).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})

    alpha_ann = model.params["const"] * ANN
    out = {
        "alpha_ann": float(alpha_ann),
        "alpha_tstat": float(model.tvalues["const"]),
        "alpha_pvalue": float(model.pvalues["const"]),
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
        "betas": {},
    }
    for c in X_.columns:
        if c == "const":
            continue
        out["betas"][c] = {
            "beta": float(model.params[c]),
            "tstat": float(model.tvalues[c]),
            "pvalue": float(model.pvalues[c]),
        }
    return out


def rolling_factor_betas(
    returns: pd.Series,
    factors: pd.DataFrame,
    window: int = 126,
    rf_col: str = "RF",
) -> pd.DataFrame:
    """Rolling OLS betas (and annualized alpha) over a trailing window.

    Returns date x [alpha_ann, <factor betas...>].
    """
    df = factors.reindex(returns.index)
    y = returns.copy()
    if rf_col in df.columns:
        y = y - df[rf_col]
        X = df.drop(columns=[rf_col])
    else:
        X = df
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(data) < window + 10:
        return pd.DataFrame()

    yv = data["y"].values
    Xv = np.column_stack([np.ones(len(data)), data.drop(columns=["y"]).values])
    cols = ["alpha_ann"] + list(data.drop(columns=["y"]).columns)
    rows, idx = [], []
    step = max(1, window // 25)  # ~25 refits per window span keeps it fast
    for end in range(window, len(data) + 1, step):
        Xw, yw = Xv[end - window:end], yv[end - window:end]
        try:
            coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        rows.append([coef[0] * ANN] + list(coef[1:]))
        idx.append(data.index[end - 1])
    return pd.DataFrame(rows, index=idx, columns=cols)


def factor_exposure_heatmap_data(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    lookback: int = 252,
    rf_col: str = "RF",
) -> pd.DataFrame:
    """Per-name factor betas (trailing regression) weighted by current portfolio
    weights -> contribution of each holding to each factor exposure.

    Returns ticker x factor matrix for the latest date (top 25 abs weights).
    """
    last_w = weights.iloc[-1]
    held = last_w[last_w.abs() > 1e-6].abs().sort_values(ascending=False).head(25).index
    rets = prices[held].pct_change(fill_method=None).tail(lookback)
    df = factors.reindex(rets.index)
    X = df.drop(columns=[rf_col]) if rf_col in df.columns else df
    X = X.dropna()
    rows = {}
    Xm = np.column_stack([np.ones(len(X)), X.values])
    for t in held:
        y = rets[t].reindex(X.index)
        mask = y.notna()
        if mask.sum() < 60:
            continue
        coef, *_ = np.linalg.lstsq(Xm[mask.values], y[mask].values, rcond=None)
        rows[t] = dict(zip(X.columns, coef[1:]))
    beta_df = pd.DataFrame(rows).T
    return beta_df.mul(last_w.reindex(beta_df.index), axis=0)
