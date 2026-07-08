"""Settings — everything customizable, in one place, persisted to the workspace.

Universe taxonomy, reusable signal definitions, cycle phases & cat events,
engine/display parameters, and workspace export/import.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from quantdash.data import get_source
from quantdash.data.insurance_events import CAT_EVENTS, MARKET_PHASES
from quantdash.data.universe import SUBSECTOR, effective_subsector
from quantdash.workspace import WS_PATH, load_workspace, save_workspace
from quantdash.ui import inject_css, page_header

st.set_page_config(page_title="Settings — Insurance Alpha Lab", page_icon="⚙️",
                   layout="wide")
inject_css()
page_header("Settings", "Taxonomy · definitions · events · parameters — "
                        "all persisted to your workspace")

ws = load_workspace()

tab_uni, tab_def, tab_ev, tab_par, tab_ws = st.tabs(
    ["Universe & Subsectors", "Signal Definitions", "Cycle & Events",
     "Parameters", "Workspace File"])

# ---------------- Universe & subsectors ----------------
with tab_uni:
    st.markdown(
        "The taxonomy drives the **Insurance universe**, **subsector "
        "neutralization**, and the **Sector Lens**. Edit assignments, add "
        "tickers with a subsector, or remove rows. New tickers also need "
        "prices — seed them in Data Explorer.")
    current = effective_subsector(ws)
    tax_df = pd.DataFrame(sorted(current.items()),
                          columns=["ticker", "subsector"])
    edited = st.data_editor(
        tax_df, num_rows="dynamic", width="stretch", height=420,
        column_config={
            "ticker": st.column_config.TextColumn(required=True),
            "subsector": st.column_config.TextColumn(
                required=True, help="Free text — new subsector names create "
                                    "new groups everywhere automatically."),
        }, key="tax_editor")
    c1, c2, c3 = st.columns(3)
    if c1.button("💾 Save taxonomy", type="primary"):
        new_map = {str(r["ticker"]).strip().upper(): str(r["subsector"]).strip()
                   for _, r in edited.iterrows()
                   if str(r["ticker"]).strip() and str(r["subsector"]).strip()}
        ws["subsector_map"] = new_map
        save_workspace(ws)
        st.success(f"Saved {len(new_map)} tickers across "
                   f"{len(set(new_map.values()))} subsectors.")
    if c2.button("↩ Reset to built-in"):
        ws["subsector_map"] = None
        save_workspace(ws)
        st.rerun()
    seeded = set(get_source().available_tickers())
    missing = [t for t in current if t not in seeded]
    if missing:
        c3.caption(f"⚠️ {len(missing)} taxonomy tickers have no price data: "
                   + ", ".join(missing[:10])
                   + ("…" if len(missing) > 10 else ""))

# ---------------- Signal definitions ----------------
with tab_def:
    st.markdown(
        "Reusable building blocks for the DSL. Each definition is evaluated "
        "before your expression and injected as a variable — later rows can "
        "reference earlier ones. Example: define `QUALITY` as "
        "`rank(-vol(63)) + rank(momentum(252, 21))`, then write "
        "`0.6*QUALITY + 0.4*rank(-drawdown(252))` anywhere a signal goes "
        "(Backtest Lab, Sweep, Walk-Forward).")
    defs = ws.get("definitions") or {}
    def_df = pd.DataFrame(
        [{"name": k, "expression": v} for k, v in defs.items()]
        or [{"name": "", "expression": ""}])
    edited_defs = st.data_editor(
        def_df, num_rows="dynamic", width="stretch",
        column_config={
            "name": st.column_config.TextColumn(
                help="Must be a valid identifier; UPPERCASE recommended."),
            "expression": st.column_config.TextColumn(width="large"),
        }, key="def_editor")
    if st.button("💾 Save definitions", type="primary", key="save_defs"):
        new_defs = {}
        bad = []
        for _, r in edited_defs.iterrows():
            name, expr = str(r["name"]).strip(), str(r["expression"]).strip()
            if not name and not expr:
                continue
            if not name.isidentifier():
                bad.append(name or "(blank)")
                continue
            new_defs[name] = expr
        ws["definitions"] = new_defs
        save_workspace(ws)
        if bad:
            st.warning("Skipped invalid names: " + ", ".join(bad))
        st.success(f"Saved {len(new_defs)} definitions.")

# ---------------- Cycle & events ----------------
with tab_ev:
    st.markdown("Used by the **cycle overlay** (Overview) and the **cat event "
                "study** (Sector Lens).")
    st.subheader("Market phases")
    phases_now = ws.get("market_phases") or [list(p) for p in MARKET_PHASES]
    ph_df = pd.DataFrame(phases_now, columns=["start", "end", "label", "kind"])
    edited_ph = st.data_editor(
        ph_df, num_rows="dynamic", width="stretch",
        column_config={"kind": st.column_config.SelectboxColumn(
            options=["soft", "hard", "transition"], required=True)},
        key="phase_editor")
    st.subheader("Cat / shock events")
    events_now = ws.get("cat_events") or [list(e) for e in CAT_EVENTS]
    ev_df = pd.DataFrame(events_now, columns=["date", "label"])
    edited_ev = st.data_editor(ev_df, num_rows="dynamic", width="stretch",
                               key="event_editor")
    c1, c2 = st.columns(2)
    if c1.button("💾 Save phases & events", type="primary"):
        ws["market_phases"] = [
            [str(r["start"]), str(r["end"]), str(r["label"]), str(r["kind"])]
            for _, r in edited_ph.iterrows() if str(r["label"]).strip()]
        ws["cat_events"] = [
            [str(r["date"]), str(r["label"])]
            for _, r in edited_ev.iterrows() if str(r["label"]).strip()]
        save_workspace(ws)
        st.success(f"Saved {len(ws['market_phases'])} phases and "
                   f"{len(ws['cat_events'])} events.")
    if c2.button("↩ Reset to built-in", key="reset_events"):
        ws["market_phases"] = None
        ws["cat_events"] = None
        save_workspace(ws)
        st.rerun()

# ---------------- Parameters ----------------
with tab_par:
    p = ws.get("params", {})
    st.markdown("Engine and display parameters used across pages.")
    aum_raw = st.text_input(
        "Capacity AUM grid ($M, comma-separated)",
        value=p.get("capacity_aum_m", "10, 50, 100, 250, 500, 1000"))
    horizons_raw = st.text_input(
        "Signal decay horizons (days, comma-separated)",
        value=p.get("decay_horizons", "1, 5, 10, 21, 63"))
    if st.button("💾 Save parameters", type="primary"):
        ws.setdefault("params", {})
        ws["params"]["capacity_aum_m"] = aum_raw
        ws["params"]["decay_horizons"] = horizons_raw
        save_workspace(ws)
        st.success("Saved. Applied on the next backtest run.")
    st.caption("Portfolio defaults (mode, costs, quantile, OOS fraction, "
               "benchmark) are saved from the Backtest Lab sidebar — "
               "'Save these settings as my defaults'.")

# ---------------- Workspace file ----------------
with tab_ws:
    st.markdown(
        "The entire configuration — presets, universes, taxonomy, definitions, "
        "events, parameters, defaults, trial history — lives in one JSON file. "
        "Export it here and import it on another machine (e.g. home ↔ desk).")
    st.download_button(
        "⬇ Export workspace.json",
        json.dumps(ws, indent=2, default=str).encode(),
        "workspace.json", "application/json")
    up = st.file_uploader("Import workspace.json", type=["json"])
    merge = st.radio("Import mode", ["Merge into current", "Replace current"],
                     horizontal=True)
    if up is not None and st.button("Import", type="primary"):
        try:
            incoming = json.loads(up.read())
            assert isinstance(incoming, dict)
        except (json.JSONDecodeError, AssertionError):
            st.error("Not a valid workspace JSON.")
            st.stop()
        if merge == "Replace current":
            save_workspace(incoming)
        else:
            for k, v in incoming.items():
                if isinstance(v, dict) and isinstance(ws.get(k), dict):
                    ws[k].update(v)
                else:
                    ws[k] = v
            save_workspace(ws)
        st.success("Imported — reloading.")
        st.rerun()
    st.caption(f"Workspace file: `{WS_PATH}`")
