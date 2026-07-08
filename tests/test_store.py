import numpy as np
import pandas as pd
import pytest

from quantdash.data.source import DuckDBSource
from quantdash.engine.attribution import position_contribution
from quantdash.engine.backtest import BacktestConfig, run_backtest
from quantdash.engine.signals import evaluate_signal


@pytest.fixture()
def store(tmp_path):
    return DuckDBSource(tmp_path / "test.duckdb")


def _price_rows(prices):
    long = prices.stack().rename("adj_close").reset_index()
    long.columns = ["date", "ticker", "adj_close"]
    for c in ["open", "high", "low", "close"]:
        long[c] = long["adj_close"]
    long["volume"] = 1e6
    long["date"] = long["date"].dt.date
    return long


def test_price_roundtrip(store, prices):
    n = store.write_prices(_price_rows(prices.iloc[:, :5]))
    assert n == 900 * 5
    panel = store.get_price_panel(["T00", "T01"])
    assert panel.shape[1] == 2
    assert np.allclose(panel["T00"].values, prices["T00"].values)
    # idempotent re-seed
    store.write_prices(_price_rows(prices.iloc[:, :5]))
    assert len(store.available_tickers()) == 5


def test_factor_and_macro_roundtrip(store, macro_panel):
    long = macro_panel.reset_index().melt(id_vars="index", var_name="series",
                                          value_name="value")
    long["date"] = long["index"].dt.date
    assert store.write_macro(long[["date", "series", "value"]]) > 0
    assert store.available_macro() == ["HY_OAS", "RATE_10Y"]
    panel = store.get_macro_panel(["HY_OAS"])
    assert list(panel.columns) == ["HY_OAS"]

    flong = long.rename(columns={"series": "factor"})
    assert store.write_factors(flong[["date", "factor", "value"]]) > 0
    assert store.available_factors() == ["HY_OAS", "RATE_10Y"]


def test_theory_roundtrip(store):
    tid = store.save_theory("t1", "hypothesis", "rank(x)", {"a": 1},
                            {"sharpe": 1.2}, "supported")
    df = store.list_theories()
    assert len(df) == 1 and df.iloc[0]["id"] == tid
    store.delete_theory(tid)
    assert store.list_theories().empty


def test_snapshot_roundtrip(store):
    rets = pd.Series([0.01, -0.005],
                     index=pd.to_datetime(["2025-01-02", "2025-01-03"]))
    store.save_snapshot("snap", "expr", {"mode": "long_short"},
                        {"sharpe": 0.9}, rets)
    snaps = store.list_snapshots()
    assert "snap" in snaps
    assert np.allclose(snaps["snap"]["returns"].values, rets.values)
    # same-name save replaces
    store.save_snapshot("snap", "expr2", {}, {}, rets)
    assert len(store.list_snapshots()) == 1
    store.delete_snapshot("snap")
    assert store.list_snapshots() == {}


def test_position_contribution_alignment(prices):
    """Regression test: weights index (live period) shorter than prices index."""
    sig = evaluate_signal("rank(momentum(126, 5))", prices)
    res = run_backtest(prices, sig, BacktestConfig())
    pc = position_contribution(res.weights, prices)  # no period bounds
    assert not pc.empty
    assert (pc["days_held"] > 0).all()
