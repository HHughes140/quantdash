"""Walk-forward analysis: re-select the best parameter in each training window,
trade it in the following test window, stitch the out-of-sample segments.

Because the DSL backtester has no path dependence across days (weights depend
only on the signal at each rebalance), each candidate parameter's full-period
daily return series can be computed once; walk-forward selection then just
slices those series. Parameter switches at window boundaries are assumed
costless (a small optimistic bias — noted in the UI).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .signals import evaluate_signal

ANN = 252


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(ANN))


def candidate_returns(
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame],
    template: str,
    a_values: Sequence,
    cfg: BacktestConfig,
    macro_panel: Optional[pd.DataFrame] = None,
    progress=None,
    definitions: Optional[dict] = None,
) -> pd.DataFrame:
    """Full-period net daily returns for each candidate parameter value."""
    out = {}
    for i, a in enumerate(a_values):
        expr = template.replace("{A}", str(a))
        sig = evaluate_signal(expr, prices, volume, macro_panel, definitions)
        res = run_backtest(prices, sig, cfg)
        out[a] = res.net_returns
        if progress:
            progress((i + 1) / len(a_values), a)
    return pd.DataFrame(out).dropna(how="all")


def walk_forward(
    cand: pd.DataFrame,
    train_days: int = 504,
    test_days: int = 126,
) -> tuple[pd.Series, pd.DataFrame]:
    """Stitch OOS returns by picking the best-train-Sharpe parameter per window.

    cand: date x parameter daily net returns (from candidate_returns).
    Returns (stitched OOS return series, per-window detail DataFrame).
    """
    idx = cand.index
    if len(idx) <= train_days + 20:
        raise ValueError(
            f"Only {len(idx)} days of candidate returns — need more than "
            f"train window ({train_days}) + 20.")

    oos_parts, rows = [], []
    pos = train_days
    while pos < len(idx):
        train = cand.iloc[pos - train_days:pos]
        test = cand.iloc[pos:pos + test_days]
        if len(test) < 10:
            break
        train_sharpes = train.apply(_sharpe)
        if train_sharpes.isna().all():
            pos += test_days
            continue
        a_star = train_sharpes.idxmax()
        oos = test[a_star]
        oos_parts.append(oos)
        test_sharpes = test.apply(_sharpe)
        rows.append({
            "window_start": test.index[0],
            "window_end": test.index[-1],
            "chosen_param": a_star,
            "train_sharpe": float(train_sharpes[a_star]),
            "oos_sharpe": _sharpe(oos),
            "oos_return": float((1 + oos.fillna(0)).prod() - 1),
            "best_hindsight_param": test_sharpes.idxmax()
            if not test_sharpes.isna().all() else None,
            "best_hindsight_sharpe": float(test_sharpes.max())
            if not test_sharpes.isna().all() else np.nan,
        })
        pos += test_days

    if not oos_parts:
        raise ValueError("No walk-forward windows could be formed.")
    stitched = pd.concat(oos_parts)
    stitched = stitched[~stitched.index.duplicated()]
    return stitched, pd.DataFrame(rows)
