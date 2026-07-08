"""Factor overlay analytics: full-sample and rolling factor regressions.

Uses snowflake_utilities.read_sql() for Axioma factor returns and optional
factor exposure pulls.

Main use:
- Regress exported L/S strategy daily returns on Snowflake Axioma factor returns.
- Run rolling factor beta regressions.
- Build factor exposure heatmaps from either exported CSVs or live Snowflake exposure data.

Important:
- Strategy signals should already be z-scored upstream in ff32.py:
    raw factor -> _Z within book/date universe -> _ZZ within book/date universe
- This module does not rebuild the L/S strategy. It analyzes the exported returns.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .snowflake_utilities import read_sql

ANN = 252

AXIOMA_FACTOR_NAMES = [
    "DIVIDEND_YIELD",
    "EARNINGS_YIELD",
    "EXCHANGE_RATE_SENSITIVITY",
    "GROWTH",
    "LEVERAGE",
    "LIQUIDITY",
    "MARKET_SENSITIVITY",
    "MEDIUM_TERM_MOMENTUM",
    "PROFITABILITY",
    "SHORT_TERM_MOMENTUM",
    "SIZE",
    "VALUE",
    "VOLATILITY",
]


def _to_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _prep_returns(s: pd.Series) -> pd.Series:
    out = _to_series(s.copy())
    out.index = pd.to_datetime(out.index, errors="coerce")
    return out[~out.index.isna()].dropna().sort_index()


def _prep_factors(factors: pd.DataFrame) -> pd.DataFrame:
    f = factors.copy()
    f.index = pd.to_datetime(f.index, errors="coerce")
    f = f[~f.index.isna()]
    for c in f.columns:
        f[c] = _to_series(f[c])
    return f.sort_index()


def sql_literal_list(values: Iterable[str]) -> str:
    vals = sorted({str(v).replace("'", "''") for v in values if pd.notna(v) and str(v).strip()})
    return ", ".join([f"'{v}'" for v in vals]) if vals else "''"


def clean_columns_upper(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).upper() for c in out.columns]
    return out


def newey_west_lags(n: int) -> int:
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def zscore_by_date(
    df: pd.DataFrame,
    value_col: str,
    out_col: str,
    date_col: str = "DDATE",
    min_names: int = 5,
) -> pd.DataFrame:
    """Cross-sectional z-score within each date universe."""
    out = df.copy()
    values = _to_series(out[value_col])
    grouped = values.groupby(out[date_col])
    count = grouped.transform("count")
    mean = grouped.transform("mean")
    std = grouped.transform(lambda x: x.std(ddof=0))
    z = (values - mean) / std.replace(0, np.nan)
    out[out_col] = z.where(count >= min_names)
    return out


def load_axioma_factor_returns_from_snowflake(
    history_days: int = 3650,
    factor_names: Sequence[str] = AXIOMA_FACTOR_NAMES,
) -> pd.DataFrame:
    """Fetch Axioma style factor returns from Snowflake.

    Confirmed schema:
        Table: AXIOMA.FUNDAMENTAL.FACTOR_RETURN
        Date column: DDATE
        Factor column: FACTORNAME
        Return column: RETURN

    Axioma returns are stored in percent units, so divide by 100.
    """
    factor_sql = sql_literal_list(factor_names)

    sql = f"""
    SELECT
        DDATE,
        FACTORNAME,
        TRY_TO_DOUBLE(RETURN) / 100.0 AS AXIOMA_FACTOR_RETURN
    FROM AXIOMA.FUNDAMENTAL.FACTOR_RETURN
    WHERE RISK_MODEL = 'WW4'
      AND HORIZON = 'SH'
      AND DDATE >= DATEADD(DAY, -{int(history_days)}, CURRENT_DATE)
      AND FACTORNAME IN ({factor_sql})
      AND RETURN IS NOT NULL
    ORDER BY DDATE, FACTORNAME
    """

    df = clean_columns_upper(read_sql(query=sql))

    if df.empty:
        return pd.DataFrame()

    required = {"DDATE", "FACTORNAME", "AXIOMA_FACTOR_RETURN"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["DDATE"] = pd.to_datetime(df["DDATE"], errors="coerce")
    df["AXIOMA_FACTOR_RETURN"] = _to_series(df["AXIOMA_FACTOR_RETURN"])
    df = df.dropna(subset=["DDATE", "FACTORNAME", "AXIOMA_FACTOR_RETURN"])

    if df.empty:
        return pd.DataFrame()

    mat = df.pivot_table(
        index="DDATE",
        columns="FACTORNAME",
        values="AXIOMA_FACTOR_RETURN",
        aggfunc="mean",
    ).sort_index()

    return _prep_factors(mat)


def load_axioma_exposures_from_snowflake(
    trading_tickers: Sequence[str],
    history_days: int = 3650,
    factor_names: Sequence[str] = AXIOMA_FACTOR_NAMES,
) -> pd.DataFrame:
    """Fetch raw Axioma factor exposures and z-score them within date universe.

    Output columns:
    DDATE, TRADING_TICKER, FACTORNAME, FACTOR_EXPOSURE, FACTOR_Z
    """
    ticker_sql = sql_literal_list(trading_tickers)
    factor_sql = sql_literal_list(factor_names)

    sql = f"""
    SELECT
        e.DDATE,
        jm.TRADING_TICKER,
        e.FACTORNAME,
        TRY_TO_DOUBLE(e.FACTOR_EXPOSURE) AS FACTOR_EXPOSURE
    FROM DB_TEAM_WILHELM_001.PUBLIC.JOINMASTER jm
    JOIN AXIOMA.FUNDAMENTAL.FACTOR_EXPOSURE e
      ON e.AXIOMAID = jm.AXIOMAID
    WHERE e.RISK_MODEL = 'WW4'
      AND e.HORIZON = 'SH'
      AND e.DDATE >= DATEADD(DAY, -{int(history_days)}, CURRENT_DATE)
      AND jm.TRADING_TICKER IN ({ticker_sql})
      AND e.FACTORNAME IN ({factor_sql})
    ORDER BY e.DDATE, jm.TRADING_TICKER, e.FACTORNAME
    """

    df = clean_columns_upper(read_sql(query=sql))
    if df.empty:
        return df

    df["DDATE"] = pd.to_datetime(df["DDATE"], errors="coerce")
    df["FACTOR_EXPOSURE"] = _to_series(df["FACTOR_EXPOSURE"])
    df = df.dropna(subset=["DDATE", "TRADING_TICKER", "FACTORNAME"])

    parts = []
    for factor, g in df.groupby("FACTORNAME", sort=False):
        z = zscore_by_date(g, "FACTOR_EXPOSURE", "FACTOR_Z", date_col="DDATE", min_names=5)
        parts.append(z)

    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    rf_col: str = "RF",
    min_obs: int = 60,
) -> dict:
    """Newey-West OLS of strategy excess returns on factor returns."""
    y = _prep_returns(returns)
    X = _prep_factors(factors)

    data = pd.concat([y.rename("strategy_return"), X], axis=1).dropna()
    if data.empty or len(data) < min_obs:
        return {"error": "not enough overlapping observations", "n_obs": int(len(data))}

    y = data["strategy_return"]
    X = data.drop(columns=["strategy_return"])

    if rf_col in X.columns:
        y = y - X[rf_col]
        X = X.drop(columns=[rf_col])

    X = X.loc[:, X.notna().sum() >= min_obs]
    X = X.loc[:, X.std(ddof=0) > 0]

    if X.empty:
        return {"error": "no usable factor columns", "n_obs": int(len(data))}

    X_ = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_, missing="drop").fit(
        cov_type="HAC",
        cov_kwds={"maxlags": newey_west_lags(len(y))},
    )

    out = {
        "n_obs": int(model.nobs),
        "alpha_daily": float(model.params.get("const", np.nan)),
        "alpha_ann": float(model.params.get("const", np.nan) * ANN),
        "alpha_tstat": float(model.tvalues.get("const", np.nan)),
        "alpha_pvalue": float(model.pvalues.get("const", np.nan)),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "resid_vol_ann": float(model.resid.std(ddof=1) * math.sqrt(ANN)),
        "betas": {},
    }

    for c in X.columns:
        out["betas"][c] = {
            "beta": float(model.params.get(c, np.nan)),
            "tstat": float(model.tvalues.get(c, np.nan)),
            "pvalue": float(model.pvalues.get(c, np.nan)),
        }

    return out


def rolling_factor_betas(
    returns: pd.Series,
    factors: pd.DataFrame,
    window: int = 126,
    rf_col: str = "RF",
    min_obs: int = 60,
    step: Optional[int] = None,
) -> pd.DataFrame:
    """Rolling OLS betas and annualized alpha."""
    y = _prep_returns(returns)
    X = _prep_factors(factors)

    data = pd.concat([y.rename("strategy_return"), X], axis=1).dropna()
    if data.empty or len(data) < max(window, min_obs):
        return pd.DataFrame()

    y_all = data["strategy_return"]
    X_all = data.drop(columns=["strategy_return"])

    if rf_col in X_all.columns:
        y_all = y_all - X_all[rf_col]
        X_all = X_all.drop(columns=[rf_col])

    X_all = X_all.loc[:, X_all.std(ddof=0) > 0]
    if X_all.empty:
        return pd.DataFrame()

    step = step or max(1, window // 25)
    rows = []

    for end in range(window, len(data) + 1, step):
        y_w = y_all.iloc[end - window:end]
        X_w = X_all.iloc[end - window:end]
        w = pd.concat([y_w.rename("y"), X_w], axis=1).dropna()

        if len(w) < min_obs:
            continue

        yy = w["y"].to_numpy(dtype=float)
        XX = np.column_stack([np.ones(len(w)), w.drop(columns=["y"]).to_numpy(dtype=float)])

        try:
            coef, *_ = np.linalg.lstsq(XX, yy, rcond=None)
            pred = XX @ coef
            resid = yy - pred
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((yy - yy.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except np.linalg.LinAlgError:
            continue

        row = {
            "DDATE": w.index[-1],
            "alpha_ann": float(coef[0] * ANN),
            "r_squared": float(r2),
        }
        for name, beta in zip(w.drop(columns=["y"]).columns, coef[1:]):
            row[name] = float(beta)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("DDATE").sort_index()


def load_strategy_daily(
    path: str | Path,
    book: Optional[str] = None,
    factor: Optional[str] = None,
    return_col: str = "long_short_return",
) -> pd.Series:
    """Load one strategy return series from a master/book L/S daily CSV."""
    df = pd.read_csv(path)
    df["DDATE"] = pd.to_datetime(df["DDATE"], errors="coerce")

    if book is not None and "BOOK" in df.columns:
        df = df[df["BOOK"].astype(str).eq(str(book))]

    if factor is not None:
        col = "factor" if "factor" in df.columns else "raw_factor_column"
        df = df[df[col].astype(str).eq(str(factor))]

    if df.empty or return_col not in df.columns:
        return pd.Series(dtype=float)

    out = df.groupby("DDATE")[return_col].mean()
    return _prep_returns(out)


def build_factor_regression_scorecard(
    strategy_daily_csv: str | Path,
    output_csv: Optional[str | Path] = None,
    history_days: int = 3650,
    return_col: str = "long_short_return",
) -> pd.DataFrame:
    """Run factor regressions for every BOOK x factor strategy.

    Axioma factor returns are loaded from Snowflake using read_sql().
    """
    daily = pd.read_csv(strategy_daily_csv)
    daily["DDATE"] = pd.to_datetime(daily["DDATE"], errors="coerce")

    factor_matrix = load_axioma_factor_returns_from_snowflake(history_days=history_days)
    if factor_matrix.empty:
        raise RuntimeError("Could not load Axioma factor returns from Snowflake.")

    rows = []
    group_cols = [c for c in ["BOOK", "return_type", "raw_factor_column", "factor"] if c in daily.columns]

    for keys, g in daily.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        meta = dict(zip(group_cols, keys))
        s = g.groupby("DDATE")[return_col].mean()
        res = factor_regression(s, factor_matrix)

        row = {**meta}
        for k, v in res.items():
            if k != "betas":
                row[k] = v

        for factor_name, b in res.get("betas", {}).items():
            row[f"beta_{factor_name}"] = b["beta"]
            row[f"tstat_{factor_name}"] = b["tstat"]
            row[f"pvalue_{factor_name}"] = b["pvalue"]

        rows.append(row)

    out = pd.DataFrame(rows)
    if "alpha_tstat" in out.columns:
        out = out.sort_values(["BOOK", "alpha_tstat"], ascending=[True, False])

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)

    return out


def build_rolling_beta_panel_from_csv(
    strategy_daily_csv: str | Path,
    output_dir: str | Path,
    history_days: int = 3650,
    return_col: str = "long_short_return",
    window: int = 126,
) -> dict[str, str]:
    """Create one rolling beta CSV per BOOK x factor strategy."""
    daily = pd.read_csv(strategy_daily_csv)
    daily["DDATE"] = pd.to_datetime(daily["DDATE"], errors="coerce")

    factor_matrix = load_axioma_factor_returns_from_snowflake(history_days=history_days)
    if factor_matrix.empty:
        raise RuntimeError("Could not load Axioma factor returns from Snowflake.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    group_cols = [c for c in ["BOOK", "return_type", "raw_factor_column", "factor"] if c in daily.columns]

    for keys, g in daily.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        meta = dict(zip(group_cols, keys))
        s = g.groupby("DDATE")[return_col].mean()
        rb = rolling_factor_betas(s, factor_matrix, window=window)

        if rb.empty:
            continue

        for k, v in meta.items():
            rb.insert(0, k, v)

        label = "_".join(str(x) for x in keys)
        label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")[:120]
        path = output_dir / f"rolling_betas_{label}.csv"
        rb.to_csv(path, index=True)
        paths[label] = str(path)

    return paths


def factor_exposure_heatmap_data(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    """Ticker x factor matrix for the app's exposure heatmap.

    Estimates per-name factor betas from a trailing regression of each held
    name's returns on the factor returns, then scales by current weight —
    exposure contribution per holding. (For production Axioma exposure
    attribution, use live_zscored_exposure_heatmap_data() or
    current_factor_exposure_heatmap_data() with live/exported exposures.)
    """
    if weights is None or prices is None or factors is None:
        return pd.DataFrame()
    if weights.empty or prices.empty or factors.empty:
        return pd.DataFrame()

    latest_weights = _to_series(weights.iloc[-1].copy())
    latest_weights = latest_weights[latest_weights.abs() > 1e-6]
    if latest_weights.empty:
        return pd.DataFrame()

    latest_weights = latest_weights[latest_weights.index.isin(prices.columns)]
    if latest_weights.empty:
        return pd.DataFrame()

    latest_weights = latest_weights.reindex(
        latest_weights.abs().sort_values(ascending=False).head(top_n).index
    )

    factor_cols = [c for c in factors.columns if str(c).upper() != "RF"]
    if not factor_cols:
        return pd.DataFrame()

    X = _prep_factors(factors[factor_cols]).dropna().tail(252)
    if len(X) < 60:
        # not enough factor history for per-name betas: weight-only fallback
        out = pd.DataFrame(index=latest_weights.index)
        for col in factor_cols:
            out[col] = latest_weights
        return out

    rets = prices[latest_weights.index].pct_change(fill_method=None).reindex(X.index)
    Xm = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    rows = {}
    for t in latest_weights.index:
        y = _to_series(rets[t])
        mask = y.notna()
        if mask.sum() < 60:
            continue
        try:
            coef, *_ = np.linalg.lstsq(Xm[mask.to_numpy()], y[mask].to_numpy(dtype=float),
                                       rcond=None)
        except np.linalg.LinAlgError:
            continue
        rows[t] = dict(zip(factor_cols, coef[1:]))

    if not rows:
        return pd.DataFrame()
    beta_df = pd.DataFrame(rows).T
    return beta_df.mul(latest_weights.reindex(beta_df.index), axis=0)


def current_factor_exposure_heatmap_data(
    current_factor_dollar_exposure_detail_csv: str | Path,
    book: Optional[str] = None,
    top_n: int = 25,
) -> pd.DataFrame:
    """Ticker x factor dollar exposure matrix from exported current exposure detail CSV."""
    df = pd.read_csv(current_factor_dollar_exposure_detail_csv)
    if df.empty:
        return pd.DataFrame()

    if book is not None and "BOOK" in df.columns:
        df = df[df["BOOK"].astype(str).eq(str(book))]

    needed = {"TRADING_TICKER", "factor", "factor_dollar_exposure"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing required columns: {needed - set(df.columns)}")

    df["factor_dollar_exposure"] = _to_series(df["factor_dollar_exposure"])

    total_abs = (
        df.groupby("TRADING_TICKER")["factor_dollar_exposure"]
        .apply(lambda s: s.abs().sum())
        .sort_values(ascending=False)
    )

    keep = total_abs.head(top_n).index
    df = df[df["TRADING_TICKER"].isin(keep)]

    mat = df.pivot_table(
        index="TRADING_TICKER",
        columns="factor",
        values="factor_dollar_exposure",
        aggfunc="sum",
    ).fillna(0.0)

    return mat.loc[total_abs.loc[keep].index]


def live_zscored_exposure_heatmap_data(
    trading_tickers: Sequence[str],
    notionals: pd.Series,
    history_days: int = 30,
    factor_names: Sequence[str] = AXIOMA_FACTOR_NAMES,
    latest_only: bool = True,
    top_n: int = 25,
) -> pd.DataFrame:
    """Live Snowflake exposure heatmap using read_sql().

    Pulls raw Axioma exposures, z-scores them within date universe, then multiplies
    latest z-score by supplied net notionals.

    notionals: index=ticker, value=net USD notional.
    """
    exp = load_axioma_exposures_from_snowflake(
        trading_tickers=trading_tickers,
        history_days=history_days,
        factor_names=factor_names,
    )
    if exp.empty:
        return pd.DataFrame()

    if latest_only:
        latest_date = exp["DDATE"].max()
        exp = exp[exp["DDATE"].eq(latest_date)].copy()

    n = _to_series(notionals)
    exp["NET_USD_NOTIONAL"] = exp["TRADING_TICKER"].map(n)
    exp["FACTOR_DOLLAR_EXPOSURE"] = _to_series(exp["NET_USD_NOTIONAL"]) * _to_series(exp["FACTOR_Z"])

    total_abs = (
        exp.groupby("TRADING_TICKER")["FACTOR_DOLLAR_EXPOSURE"]
        .apply(lambda s: s.abs().sum())
        .sort_values(ascending=False)
    )

    keep = total_abs.head(top_n).index
    exp = exp[exp["TRADING_TICKER"].isin(keep)]

    mat = exp.pivot_table(
        index="TRADING_TICKER",
        columns="FACTORNAME",
        values="FACTOR_DOLLAR_EXPOSURE",
        aggfunc="sum",
    ).fillna(0.0)

    return mat.loc[total_abs.loc[keep].index]
