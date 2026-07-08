import json

import pytest

import quantdash.workspace as wsmod
from quantdash.data.insurance_events import (CAT_EVENTS, MARKET_PHASES,
                                             get_cat_events, get_market_phases)
from quantdash.data.universe import SUBSECTOR, effective_subsector
from quantdash.engine.signals import evaluate_signal


def test_definitions_inject_and_chain(prices):
    defs = {
        "QUALITY": "rank(-vol(63))",
        "COMBO": "QUALITY + rank(momentum(126, 5))",
    }
    sig = evaluate_signal("0.5 * COMBO", prices, definitions=defs)
    assert sig.shape == prices.shape
    assert sig.notna().any().any()


def test_definition_error_names_the_definition(prices):
    with pytest.raises(ValueError, match="Definition 'BAD'"):
        evaluate_signal("BAD", prices, definitions={"BAD": "nonexistent(1)"})


def test_invalid_definition_names_skipped(prices):
    sig = evaluate_signal("rank(returns(21))", prices,
                          definitions={"not valid!": "vol(5)", "": "x"})
    assert sig.shape == prices.shape


def test_effective_subsector_override():
    assert effective_subsector(None) == SUBSECTOR
    ws = {"subsector_map": {"pgr": "My Group", "NEWCO": "Startups"}}
    m = effective_subsector(ws)
    assert m == {"PGR": "My Group", "NEWCO": "Startups"}


def test_event_getters_fallback_and_override():
    assert get_market_phases({}) == MARKET_PHASES
    assert get_cat_events(None) == CAT_EVENTS
    ws = {"market_phases": [["2020-01-01", "2021-01-01", "Custom", "hard"]],
          "cat_events": [["2020-03-11", "COVID"]]}
    assert get_market_phases(ws) == [("2020-01-01", "2021-01-01", "Custom", "hard")]
    assert get_cat_events(ws) == [("2020-03-11", "COVID")]


def test_workspace_preserves_unknown_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(wsmod, "WS_PATH", tmp_path / "ws.json")
    ws = wsmod.load_workspace()
    ws["future_feature"] = {"x": 1}
    ws["definitions"]["Q"] = "rank(vol(5))"
    wsmod.save_workspace(ws)
    ws2 = wsmod.load_workspace()
    assert ws2["future_feature"] == {"x": 1}
    assert ws2["definitions"]["Q"] == "rank(vol(5))"
    assert ws2["defaults"]["mode"] == "long_short"  # defaults still merged


def test_workspace_corrupt_file_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "ws.json"
    path.write_text("{not json")
    monkeypatch.setattr(wsmod, "WS_PATH", path)
    ws = wsmod.load_workspace()
    assert ws["signal_presets"] == {}
