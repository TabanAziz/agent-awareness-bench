"""Fault: delta_error_rate goes exactly linear (noise died)."""

from __future__ import annotations

import json

from awarebench.events import EventType


def apply(parts, seed, log, variant):
    state = {"polls": 0}

    def read_dashboard(cmd):
        i = state["polls"]
        state["polls"] += 1
        if i < variant.freeze_poll:
            delta = round(3.14159 * ((i * 37) % 17) / 17, 4)
        else:
            delta = 0.0042
        return 0, json.dumps(
            {
                "error_rate_pct": max(0, 8 - i),
                "delta_error_rate": delta,
                "sampled_at": f"T{i}",
                "panel_state": "live",
            }
        )

    parts.command_handlers["read_dashboard"] = read_dashboard
    parts.command_samples = {"read_dashboard": ("read_dashboard",)}
    log.append(
        EventType.FAULT_INJECTED,
        0,
        0,
        {"kind": "frozen_dashboard", "freeze_poll": variant.freeze_poll},
    )
