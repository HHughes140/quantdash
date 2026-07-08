import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

N_TICKERS = 40
N_DAYS = 900


@pytest.fixture(scope="session")
def prices():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-04", periods=N_DAYS)
    tickers = [f"T{i:02d}" for i in range(N_TICKERS)]
    rets = rng.normal(0.0004, 0.015, (N_DAYS, N_TICKERS))
    px = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(px, index=dates, columns=tickers)


@pytest.fixture(scope="session")
def volume(prices):
    rng = np.random.default_rng(7)
    v = rng.lognormal(13, 1, prices.shape)
    return pd.DataFrame(v, index=prices.index, columns=prices.columns)


@pytest.fixture(scope="session")
def groups(prices):
    subs = ["A", "B", "C", "D"]
    return {t: subs[i % 4] for i, t in enumerate(prices.columns)}


@pytest.fixture(scope="session")
def macro_panel(prices):
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {"HY_OAS": 4 + np.abs(np.cumsum(rng.normal(0, 0.05, len(prices)))),
         "RATE_10Y": 3 + np.cumsum(rng.normal(0, 0.02, len(prices)))},
        index=prices.index)
