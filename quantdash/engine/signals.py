"""Signal expression DSL.

A signal expression is a one-line Python expression evaluated over a price
panel (date x ticker). It must produce a DataFrame of the same shape; higher
value = more attractive. Cross-sectional and time-series helpers are provided
in the namespace, so theories read like:

    rank(returns(252) - returns(21))          # 12-1 momentum
    -zscore(returns(5))                       # short-term reversal
    -vol(63)                                  # low volatility
    rank(returns(126)) / (1 + vol(21))        # vol-adjusted momentum
    zscore(volume_ratio(5, 63))               # volume shock

Macro conditioning (series uploaded via Data Explorer):

    where(macro("HY_OAS") > 5, rank(-vol(63)), rank(momentum(252, 21)))
    rank(momentum(252, 21)) * sign(macro_chg("PC_PRICING_INDEX", 63))
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_namespace(prices: pd.DataFrame, volume: pd.DataFrame | None,
                    macro_panel: pd.DataFrame | None = None):
    daily = prices.pct_change(fill_method=None)

    def _macro_series(name: str) -> pd.Series:
        if macro_panel is None or macro_panel.empty:
            raise ValueError(
                f"macro('{name}') used but no macro data is loaded — upload a "
                "workbook in Data Explorer first")
        if name not in macro_panel.columns:
            raise ValueError(
                f"macro series '{name}' not found. Available: "
                + ", ".join(sorted(macro_panel.columns)))
        return macro_panel[name].reindex(prices.index).ffill()

    def _broadcast(s: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile(s.to_numpy()[:, None], (1, len(prices.columns))),
            index=prices.index, columns=prices.columns)

    def macro(name: str) -> pd.DataFrame:
        """Macro series level, forward-filled and broadcast across tickers."""
        return _broadcast(_macro_series(name))

    def macro_z(name: str, n: int = 252) -> pd.DataFrame:
        """Trailing n-day z-score of a macro series (broadcast)."""
        s = _macro_series(name)
        z = (s - s.rolling(n).mean()) / s.rolling(n).std()
        return _broadcast(z)

    def macro_chg(name: str, n: int = 21) -> pd.DataFrame:
        """n-day change (difference) in a macro series (broadcast)."""
        return _broadcast(_macro_series(name).diff(n))

    # ---- time-series operators (per ticker) ----
    def returns(n: int) -> pd.DataFrame:
        """Trailing n-day total return."""
        return prices.pct_change(n, fill_method=None)

    def log_returns(n: int) -> pd.DataFrame:
        return np.log(prices / prices.shift(n))

    def momentum(lookback: int = 252, skip: int = 21) -> pd.DataFrame:
        """Classic momentum: return over `lookback` days excluding most recent `skip`."""
        return prices.shift(skip).pct_change(lookback - skip, fill_method=None)

    def vol(n: int = 63) -> pd.DataFrame:
        """Annualized rolling volatility of daily returns."""
        return daily.rolling(n).std() * np.sqrt(252)

    def sma(n: int) -> pd.DataFrame:
        return prices.rolling(n).mean()

    def ema(n: int) -> pd.DataFrame:
        return prices.ewm(span=n, adjust=False).mean()

    def price() -> pd.DataFrame:
        return prices

    def delay(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.shift(n)

    def delta(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.diff(n)

    def ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Rolling percentile rank of today's value within the last n days."""
        return df.rolling(n).rank(pct=True)

    def ts_zscore(df: pd.DataFrame, n: int) -> pd.DataFrame:
        m, s = df.rolling(n).mean(), df.rolling(n).std()
        return (df - m) / s

    def drawdown(n: int = 252) -> pd.DataFrame:
        """Distance from rolling n-day high (negative numbers)."""
        return prices / prices.rolling(n).max() - 1.0

    def rsi(n: int = 14) -> pd.DataFrame:
        chg = prices.diff()
        gain = chg.clip(lower=0).rolling(n).mean()
        loss = (-chg.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def volume_ratio(short: int = 5, long: int = 63) -> pd.DataFrame:
        """Short-window average volume relative to long-window average."""
        if volume is None or volume.empty:
            raise ValueError("volume data not available for this universe")
        return volume.rolling(short).mean() / volume.rolling(long).mean()

    # ---- cross-sectional operators (per date) ----
    def rank(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional percentile rank in [0, 1] each day."""
        return df.rank(axis=1, pct=True)

    def zscore(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean(axis=1)
        sd = df.std(axis=1)
        return df.sub(mu, axis=0).div(sd, axis=0)

    def demean(df: pd.DataFrame) -> pd.DataFrame:
        return df.sub(df.mean(axis=1), axis=0)

    def winsorize(df: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
        zs = zscore(df)
        return zs.clip(-z, z)

    ns = {
        "returns": returns, "log_returns": log_returns, "momentum": momentum,
        "vol": vol, "sma": sma, "ema": ema, "price": price, "delay": delay,
        "delta": delta, "ts_rank": ts_rank, "ts_zscore": ts_zscore,
        "drawdown": drawdown, "rsi": rsi, "volume_ratio": volume_ratio,
        "rank": rank, "zscore": zscore, "demean": demean, "winsorize": winsorize,
        "macro": macro, "macro_z": macro_z, "macro_chg": macro_chg,
        "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
        "exp": np.exp, "clip": lambda df, lo, hi: df.clip(lo, hi),
        "where": lambda cond, a, b: a.where(cond, b),
        "np": np, "pd": pd,
    }
    return ns


def evaluate_signal(
    expression: str,
    prices: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    macro_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate a signal expression over the price panel.

    macro_panel: optional date x series DataFrame exposed via macro()/macro_z()/
    macro_chg() in the expression namespace.

    Returns a date x ticker DataFrame (higher = more attractive).
    Raises ValueError with a readable message on bad expressions.
    """
    if not expression or not expression.strip():
        raise ValueError("Empty signal expression")
    ns = _make_namespace(prices, volume, macro_panel)
    try:
        result = eval(expression, {"__builtins__": {}}, ns)  # noqa: S307 - local research tool
    except Exception as e:
        raise ValueError(f"Signal expression error: {type(e).__name__}: {e}") from e
    if isinstance(result, pd.Series):
        result = result.to_frame().reindex(columns=prices.columns)
    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"Expression must produce a DataFrame (date x ticker), got {type(result).__name__}"
        )
    result = result.reindex(index=prices.index, columns=prices.columns)
    return result.replace([np.inf, -np.inf], np.nan)


SIGNAL_PRESETS = {
    "Momentum 12-1": "rank(momentum(252, 21))",
    "Short-term reversal (5d)": "-zscore(returns(5))",
    "Low volatility": "-vol(63)",
    "Vol-adjusted momentum": "rank(momentum(252, 21)) / (1 + vol(63))",
    "Trend (price vs 200d SMA)": "price() / sma(200) - 1",
    "52-week high proximity": "-drawdown(252)",
    "Volume shock": "zscore(volume_ratio(5, 63))",
    "Mean reversion (RSI)": "-zscore(rsi(14))",
    "Momentum x low-vol composite": "rank(momentum(252, 21)) + rank(-vol(63))",
}
