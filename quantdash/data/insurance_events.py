"""Insurance market cycle phases and major catastrophe/market events.

Dates are approximate industry consensus, meant for chart overlays and event
studies — not for trading logic. Extend freely; the UI reads whatever is here.
"""

from __future__ import annotations

import pandas as pd

# (start, end, label, kind) — kind: "soft" | "hard" | "transition"
MARKET_PHASES = [
    ("2015-01-01", "2019-06-30", "Soft market", "soft"),
    ("2019-07-01", "2020-02-28", "Hardening", "transition"),
    ("2020-03-01", "2023-12-31", "Hard market", "hard"),
    ("2024-01-01", "2026-12-31", "Moderating / rate adequacy", "transition"),
]

# (date, label) — major cat and industry shock events
CAT_EVENTS = [
    ("2017-08-25", "Hurricane Harvey"),
    ("2017-09-10", "Hurricane Irma"),
    ("2017-09-20", "Hurricane Maria"),
    ("2018-10-10", "Hurricane Michael"),
    ("2020-03-11", "COVID-19 declared pandemic"),
    ("2021-02-15", "Winter Storm Uri"),
    ("2021-08-29", "Hurricane Ida"),
    ("2022-09-28", "Hurricane Ian"),
    ("2023-08-08", "Maui wildfires"),
    ("2024-09-26", "Hurricane Helene"),
    ("2024-10-09", "Hurricane Milton"),
    ("2025-01-07", "LA wildfires"),
]

PHASE_COLORS = {
    "soft": "rgba(139,147,167,0.10)",
    "hard": "rgba(229,180,84,0.10)",
    "transition": "rgba(91,141,239,0.07)",
}


def event_study(
    prices: pd.DataFrame,
    group_of: dict,
    events: list[tuple[str, str]] | None = None,
    pre_days: int = 10,
    post_days: int = 30,
) -> pd.DataFrame:
    """Average cumulative return by group around events.

    Returns DataFrame indexed by day offset (-pre..+post), one column per
    group: mean cumulative return across events, rebased to 0 at offset 0.
    """
    events = events or CAT_EVENTS
    rets = prices.pct_change(fill_method=None)
    groups = sorted({group_of.get(t, "Other") for t in prices.columns})
    per_event = {g: [] for g in groups}

    for date_str, _label in events:
        dt = pd.Timestamp(date_str)
        pos = rets.index.searchsorted(dt)
        if pos - pre_days < 0 or pos + post_days >= len(rets.index):
            continue
        window = rets.iloc[pos - pre_days: pos + post_days + 1]
        for g in groups:
            members = [t for t in prices.columns if group_of.get(t, "Other") == g]
            if len(members) < 2:
                continue
            eq = (1 + window[members].mean(axis=1).fillna(0)).cumprod()
            eq = eq / eq.iloc[pre_days] - 1  # rebase to 0 at the event date
            eq.index = range(-pre_days, post_days + 1)
            per_event[g].append(eq)

    out = {}
    for g, curves in per_event.items():
        if curves:
            out[g] = pd.concat(curves, axis=1).mean(axis=1)
    return pd.DataFrame(out)
