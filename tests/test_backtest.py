import numpy as np
import pandas as pd
import pytest

from quantdash.engine.backtest import BacktestConfig, run_backtest
from quantdash.engine.signals import evaluate_signal


@pytest.fixture(scope="module")
def signal(prices):
    return evaluate_signal("rank(momentum(126, 5))", prices)


def test_long_only_weights_sum_to_one(prices, signal):
    res = run_backtest(prices, signal, BacktestConfig(mode="long_only"))
    live = res.weights[res.weights.abs().sum(axis=1) > 0]
    assert np.allclose(live.sum(axis=1), 1.0, atol=1e-6)


def test_long_short_dollar_neutral(prices, signal):
    res = run_backtest(prices, signal, BacktestConfig(mode="long_short"))
    live = res.weights[res.weights.abs().sum(axis=1) > 0]
    assert np.allclose(live.sum(axis=1), 0.0, atol=1e-6)
    assert np.allclose(live.abs().sum(axis=1), 1.0, atol=1e-6)


def test_costs_reduce_returns(prices, signal):
    free = run_backtest(prices, signal, BacktestConfig(cost_bps=0.0))
    costly = run_backtest(prices, signal, BacktestConfig(cost_bps=25.0))
    assert costly.net_returns.sum() < free.net_returns.sum()
    assert np.allclose(free.gross_returns, costly.gross_returns)


def test_no_lookahead(prices, signal):
    """Weights before date T must be unchanged if prices after T change."""
    cfg = BacktestConfig()
    res_full = run_backtest(prices, signal, cfg)
    cut = prices.index[600]
    tampered = prices.copy()
    tampered.loc[tampered.index > cut] *= np.random.default_rng(1).uniform(
        0.5, 1.5, (int((tampered.index > cut).sum()), tampered.shape[1]))
    sig2 = evaluate_signal("rank(momentum(126, 5))", tampered)
    res_tampered = run_backtest(tampered, sig2, cfg)
    before = res_full.weights.loc[:cut]
    before2 = res_tampered.weights.loc[:cut]
    pd.testing.assert_frame_equal(before, before2)


def test_subsector_neutralization_reduces_group_tilt(prices, groups):
    """A signal that is purely a group bet should be flattened by neutralize."""
    rng = np.random.default_rng(3)
    group_score = {"A": 2.0, "B": 1.0, "C": -1.0, "D": -2.0}
    base = pd.DataFrame(
        {c: group_score[groups[c]] + rng.normal(0, 0.1, len(prices))
         for c in prices.columns}, index=prices.index)
    cfg_plain = BacktestConfig(mode="long_short")
    cfg_neut = BacktestConfig(mode="long_short", neutralize="subsector")
    res_plain = run_backtest(prices, base, cfg_plain, groups=groups)
    res_neut = run_backtest(prices, base, cfg_neut, groups=groups)

    def max_group_net(res):
        w = res.weights.iloc[-1]
        gsum = w.groupby(pd.Series({c: groups[c] for c in w.index})).sum()
        return gsum.abs().max()

    assert max_group_net(res_neut) < max_group_net(res_plain) * 0.5


def test_borrow_cost_hits_shorts_only(prices, signal):
    ls = run_backtest(prices, signal,
                      BacktestConfig(mode="long_short", cost_bps=0,
                                     short_borrow_annual_bps=200))
    ls_free = run_backtest(prices, signal,
                           BacktestConfig(mode="long_short", cost_bps=0))
    lo = run_backtest(prices, signal,
                      BacktestConfig(mode="long_only", cost_bps=0,
                                     short_borrow_annual_bps=200))
    lo_free = run_backtest(prices, signal,
                           BacktestConfig(mode="long_only", cost_bps=0))
    assert ls.net_returns.sum() < ls_free.net_returns.sum()
    assert np.isclose(lo.net_returns.sum(), lo_free.net_returns.sum())


def test_beta_hedge_reduces_beta(prices, signal):
    bench = prices.mean(axis=1)  # equal-weight index as benchmark
    plain = run_backtest(prices, signal,
                         BacktestConfig(mode="long_only"), bench)
    hedged = run_backtest(prices, signal,
                          BacktestConfig(mode="long_only", beta_hedge=True),
                          bench)

    def beta(res):
        b = res.benchmark_returns.dropna()
        r = res.net_returns.reindex(b.index)
        return abs(r.cov(b) / b.var())

    assert beta(hedged) < beta(plain) * 0.5


def test_drift_weights_differ_between_rebalances(prices, signal):
    still = run_backtest(prices, signal, BacktestConfig(rebalance_every=21))
    drift = run_backtest(prices, signal,
                         BacktestConfig(rebalance_every=21, drift_weights=True))
    # same book on effective rebalance days, different in between
    mid = still.weights.index[500]
    assert not np.allclose(still.weights.loc[mid], drift.weights.loc[mid])
    # drifted weights still finite and bounded
    assert np.isfinite(drift.weights.values).all()
    assert drift.weights.abs().values.max() < 1.0
