"""User workspace: saved signal presets, named universes, and default settings.

Stored as JSON at data/workspace.json (gitignored) so customization survives
restarts without touching code.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

WS_PATH = Path(__file__).resolve().parents[1] / "data" / "workspace.json"

_DEFAULTS = {
    "signal_presets": {},   # name -> expression
    "universes": {},        # name -> [tickers]
    "defaults": {
        "mode": "long_short",
        "quantile": 0.2,
        "rebalance": 5,
        "cost_bps": 5.0,
        "max_weight": 0.10,
        "vol_target": 0.0,
        "oos_frac": 0.3,
        "benchmark": None,
    },
}


def load_workspace() -> dict:
    ws = deepcopy(_DEFAULTS)
    if WS_PATH.exists():
        try:
            saved = json.loads(WS_PATH.read_text())
            for key in ("signal_presets", "universes"):
                if isinstance(saved.get(key), dict):
                    ws[key].update(saved[key])
            if isinstance(saved.get("defaults"), dict):
                ws["defaults"].update(saved["defaults"])
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/unreadable workspace: fall back to defaults
    return ws


def save_workspace(ws: dict) -> None:
    WS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WS_PATH.write_text(json.dumps(ws, indent=2, default=str))
