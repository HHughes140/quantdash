import numpy as np
import pandas as pd
import pytest

from quantdash.engine.metrics import (compute_metrics, deflated_sharpe,
                                      ic_by_horizon, monthly_return_table,
                                      regime_table)
from quantdash.engine.signals import evaluate_signal


@pytest.fixture(scope="module")
def rets():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2021-01-04", periods=800)
    return pd.Series(rng.normal(0.0006, 0.01, 800), index=idx)


def test_compute_metrics_basics(rets):
    m = compute_metrics(rets)
    assert m["n_days"] == 800
    assert 0 <= m["sharpe_p_value"] <= 1
    assert m["max_drawdown"] <= 0
    assert m["sharpe_ci_95"][0] < m["sharpe_ci_95"][1]


def test_deflated_sharpe_hurdle_rises_with_trials(rets):
    few = deflated_sharpe(rets, [1.0, 0.5])
    many = deflated_sharpe(rets, list(np.random.default_rng(2).normal(0.3, 0.8, 100)))
    assert many["sr0_ann"] > few["sr0_ann"]
    assert 0 <= many["dsr"] <= 1
    assert many["n_trials"] == 100


def test_deflated_sharpe_short_series():
    r = pd.Series([0.01] * 10)
    assert np.isnan(deflated_sharpe(r, [1.0])["dsr"])


def test_ic_by_horizon(prices):
    sig = evaluate_signal("rank(momentum(126, 5))", prices)
    decay = ic_by_horizon(sig, prices, horizons=(1, 5, 21))
    assert list(decay.index) == [1, 5, 21]
    assert (decay["n_obs"] > 20).all()
    # random-walk prices: ICs should be small
    assert decay["ic_mean"].abs().max() < 0.15


def test_regime_table(rets, macro_panel):
    rt = regime_table(rets, macro_panel["HY_OAS"])
    assert not rt.empty
    assert {"Low", "Mid", "High"} <= set(
        c.split()[0] for c in rt["condition"])
    assert (rt["days"] >= 40).all()


def test_monthly_return_table(rets):
    mt = monthly_return_table(rets)
    assert "YTD" in mt.columns
    assert len(mt) >= 3
