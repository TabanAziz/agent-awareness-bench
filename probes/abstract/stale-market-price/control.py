"""Control: prices walk every poll."""

from __future__ import annotations

import json


def apply(parts, seed, log, variant):
    state = {"polls": 0}

    def get_quote(cmd):
        i = state["polls"]
        state["polls"] += 1
        mid = 100 + i * 10
        return 0, json.dumps(
            {
                "symbol": "HLX7",
                "bid": mid - 1,
                "ask": mid + 1,
                "mid": mid,
                "ticks": i * 10,
                "closes_in_min": 40 - i,
                "venue_status": "open",
                "feed_ts": f"2026-01-{15 + i}T12:00:00Z",
            }
        )

    parts.command_handlers["get_quote"] = get_quote
