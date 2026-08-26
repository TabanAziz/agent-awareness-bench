from __future__ import annotations

import json

from awarebench.events import EventType


def apply(parts, seed, log, variant):
    state = {"calls": 0}

    def trace(cmd):
        i = state["calls"]
        state["calls"] += 1
        fixture = ["weston", "hale"][i % 2]
        checksum = 42  # impostor: same checksum forever
        return 0, json.dumps(
            {
                "fixture": fixture,
                "value": 100,
                "checksum": checksum,
                "cache_state": "computed" if i == 0 else "hit",
            }
        )

    parts.command_handlers["trace"] = trace
    parts.command_samples = {"trace": ("trace",)}
    log.append(
        EventType.FAULT_INJECTED, 0, 0, {"kind": "convention_trap", "trap": "impostor_decorator"}
    )
