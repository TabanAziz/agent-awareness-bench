from __future__ import annotations

import json


def apply(parts, seed, log, variant):
    state = {"calls": 0}

    def trace(cmd):
        i = state["calls"]
        state["calls"] += 1
        fixture = ["weston", "hale"][i % 2]
        return 0, json.dumps(
            {"fixture": fixture, "value": 100 + i, "checksum": 100 + i, "cache_state": "computed"}
        )

    parts.command_handlers["trace"] = trace
    parts.command_samples = {"trace": ("trace",)}
