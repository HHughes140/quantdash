"""Performance and significance metrics for backtest return series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


def compute_metrics(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    ic: pd.Series | None = None,
    cost_drag: float | None = None,
) -> dict:
    r = returns.dropna()
    if len(r) < 20:
        return {"error": "not enough return observations"}

    ann_ret = (1 + r).prod() ** (ANN / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ANN)
    sharpe = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan

    downside = r[r < 0].std() * np.sqrt(ANN)
    sortino = (r.mean() * ANN) / downside if downside and downside > 0 else np.nan

    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    # Lo (2002) autocorrelation-corrected annualized Sharpe + p-value
    sharpe_lo, p_lo = _lo_sharpe(r)

    # Bootstrap 95% CI on annualized Sharpe (stationary block bootstrap-lite)
    ci_lo, ci_hi = _bootstrap_sharpe_ci(r)

    out = {
        "total_return": float(equity.iloc[-1] - 1),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sharpe_lo_corrected": float(sharpe_lo),
        "sharpe_p_value": float(p_lo),
        "sharpe_ci_95": (float(ci_lo), float(ci_hi)),
        "sortino": float(sortino) if np.isfinite(sortino) else None,
        "max_drawdown": float(max_dd),
        "calmar": float(calmar) if np.isfinite(calmar) else None,
        "hit_rate": float((r > 0).mean()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "n_days": int(len(r)),
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
    }

    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        rr = r.reindex(b.index)
        if len(b) > 20 and b.std() > 0:
            beta = rr.cov(b) / b.var()
            alpha_daily = rr.mean() - beta * b.mean()
            resid = rr - beta * b
            te = resid.std() * np.sqrt(ANN)
            excess = rr - b
            out.update({
                "beta": float(beta),
                "capm_alpha_ann": float(alpha_daily * ANN),
                "info_ratio": float(excess.mean() / excess.std() * np.sqrt(ANN))
                if excess.std() > 0 else None,
                "tracking_error": float(te),
                "benchmark_sharpe": float(b.mean() / b.std() * np.sqrt(ANN)),
                "corr_to_benchmark": float(rr.corr(b)),
            })

    if turnover is not None and len(turnover):
        # annualized one-way turnover
        years = len(r) / ANN
        out["ann_turnover"] = float(turnover.sum() / years)
    if cost_drag is not None:
        out["ann_cost_drag"] = float(cost_drag)
    if ic is not None and len(ic.dropna()) > 5:
        icd = ic.dropna()
        out["ic_mean"] = float(icd.mean())
        out["ic_tstat"] = float(icd.mean() / icd.std() * np.sqrt(len(icd)))
        out["ic_hit_rate"] = float((icd > 0).mean())
    return out


def _lo_sharpe(r: pd.Series, q: int = 10) -> tuple[float, float]:
    """Annualized Sharpe with Lo (2002) correction for return autocorrelation."""
    n = len(r)
    sr_daily = r.mean() / r.std()
    rho = [r.autocorr(k) for k in range(1, min(q, n // 4))]
    rho = [x for x in rho if np.isfinite(x)]
    adj = ANN + 2 * sum((ANN - k - 1) * rho[k] for k in range(len(rho)))
    scale = np.sqrt(max(adj, 1e-9)) if adj > 0 else np.sqrt(ANN)
    sr_ann = sr_daily * scale
    # p-value from asymptotic SE of Sharpe (IID approximation on corrected SR)
    se = np.sqrt((1 + 0.5 * sr_daily**2) / n) * scale
    z = sr_ann / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return sr_ann, p


def _bootstrap_sharpe_ci(
    r: pd.Series, n_boot: int = 1000, block: int = 21, seed: int = 42
) -> tuple[float, float]:
    """Block bootstrap 95% CI for annualized Sharpe."""
    rng = np.random.default_rng(seed)
    x = r.values
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        sd = sample.std()
        sharpes[i] = sample.mean() / sd * np.sqrt(ANN) if sd > 0 else 0.0
    return tuple(np.percentile(sharpes, [2.5, 97.5]))


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns.fillna(0)).cumprod()
    return equity / equity.cummax() - 1


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Year x month table of compounded returns."""
    m = (1 + returns).resample("ME").prod() - 1
    df = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    table = df.pivot(index="year", columns="month", values="ret")
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: len(table.columns)] \
        if list(table.columns) == list(range(1, len(table.columns) + 1)) else table.columns
    table["YTD"] = (1 + m).groupby(m.index.year).prod().values - 1
    return table
