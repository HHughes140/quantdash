import numpy as np
import pandas as pd
import pytest

from quantdash.engine.backtest import BacktestConfig, run_backtest
from quantdash.engine.capacity import capacity_analysis, dollar_adv
from quantdash.engine.signals import evaluate_signal
from quantdash.engine.walkforward import candidate_returns, walk_forward


def test_walk_forward_windows(prices, volume):
    cand = candidate_returns(prices, volume, "rank(momentum({A}, 5))",
                             [63, 126], BacktestConfig(cost_bps=5))
    assert list(cand.columns) == [63, 126]
    wf, windows = walk_forward(cand, train_days=252, test_days=63)
    assert len(windows) >= 5
    assert set(windows["chosen_param"]) <= {63, 126}
    # stitched series covers the post-train period without duplicates
    assert not wf.index.duplicated().any()
    assert (windows["train_sharpe"].notna()).all()


def test_walk_forward_too_short_raises(prices, volume):
    cand = candidate_returns(prices.iloc[:200], volume.iloc[:200],
                             "rank(returns({A}))", [21],
                             BacktestConfig())
    with pytest.raises(ValueError, match="need more"):
        walk_forward(cand, train_days=500, test_days=63)


def test_capacity_monotone_in_aum(prices, volume):
    sig = evaluate_signal("rank(momentum(126, 5))", prices)
    res = run_backtest(prices, sig, BacktestConfig())
    adv = dollar_adv(prices, volume)
    cap = capacity_analysis(res.net_returns, res.weights, prices, adv,
                            aum_grid=(1e6, 100e6, 10e9))
    drags = cap["ann_impact_drag"].values
    assert drags[0] < drags[1] < drags[2]
    sharpes = cap["sharpe_after_impact"].values
    assert sharpes[0] > sharpes[2]
    assert 0 <= cap["median_participation"].iloc[0] <= 1
