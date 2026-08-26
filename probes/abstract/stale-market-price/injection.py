"""Fault: mid freezes while ticks/feed_ts advance."""

from __future__ import annotations

import json

from awarebench.events import EventType


def apply(parts, seed, log, variant):
    state = {"polls": 0}

    def get_quote(cmd):
        i = state["polls"]
        state["polls"] += 1
        if i < variant.freeze_poll:
            mid = 100 + i * 10
        else:
            mid = 100 + (variant.freeze_poll - 1) * 10
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
    parts.command_samples = {"get_quote": ("get_quote",)}
    log.append(
        EventType.FAULT_INJECTED,
        0,
        0,
        {"kind": "stale_market_price", "freeze_poll": variant.freeze_poll},
    )
