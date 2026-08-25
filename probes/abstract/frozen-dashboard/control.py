"""Control: noise persists throughout."""

from __future__ import annotations

import json


def apply(parts, seed, log, variant):
    state = {"polls": 0}

    def read_dashboard(cmd):
        i = state["polls"]
        state["polls"] += 1
        delta = round(3.14159 * ((i * 37 + 7) % 17) / 17, 4)
        return 0, json.dumps(
            {
                "error_rate_pct": max(0, 8 - i),
                "delta_error_rate": delta,
                "sampled_at": f"T{i}",
                "panel_state": "live",
            }
        )

    parts.command_handlers["read_dashboard"] = read_dashboard
