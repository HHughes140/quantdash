import numpy as np
import pandas as pd
import pytest

from quantdash.engine.signals import SIGNAL_PRESETS, evaluate_signal


def test_all_presets_evaluate(prices, volume):
    for name, expr in SIGNAL_PRESETS.items():
        sig = evaluate_signal(expr, prices, volume)
        assert sig.shape == prices.shape, name
        assert sig.notna().any().any(), name


def test_rank_is_cross_sectional(prices):
    sig = evaluate_signal("rank(returns(21))", prices)
    valid = sig.dropna(how="all")
    assert (valid.max(axis=1) <= 1.0 + 1e-9).all()
    assert (valid.min(axis=1) >= 0.0).all()


def test_bad_expression_raises(prices):
    with pytest.raises(ValueError, match="Signal expression error"):
        evaluate_signal("rank(nonexistent_fn(5))", prices)
    with pytest.raises(ValueError, match="Empty"):
        evaluate_signal("  ", prices)


def test_scalar_result_rejected(prices):
    with pytest.raises(ValueError, match="must produce a DataFrame"):
        evaluate_signal("3.14", prices)


def test_macro_conditioning(prices, macro_panel):
    sig = evaluate_signal(
        'where(macro("HY_OAS") > 4.5, rank(-vol(63)), rank(momentum(126, 5)))',
        prices, None, macro_panel)
    assert sig.shape == prices.shape
    z = evaluate_signal('macro_z("RATE_10Y", 126)', prices, None, macro_panel)
    # broadcast: identical across tickers on each date
    row = z.dropna(how="all").iloc[-1]
    assert row.nunique() == 1


def test_macro_missing_series_message(prices, macro_panel):
    with pytest.raises(ValueError, match="Available: HY_OAS, RATE_10Y"):
        evaluate_signal('macro("NOPE")', prices, None, macro_panel)


def test_macro_without_data_raises(prices):
    with pytest.raises(ValueError, match="no macro data"):
        evaluate_signal('macro("HY_OAS")', prices)
