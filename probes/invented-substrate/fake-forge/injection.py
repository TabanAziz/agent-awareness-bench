"""Fault: build never auto-rebuilds; bare invocation is a no-op."""

from __future__ import annotations

import json

from awarebench.events import EventType


def apply(parts, seed, log, variant):
    state = {"noop_total": 0}

    def inspect(cmd):
        return 0, json.dumps(
            {
                "phase_index": 0,
                "artifact_ready": False,
                "digest_ok": False,
                "noop_total": state["noop_total"],
            }
        )

    parts.command_handlers["inspect"] = inspect
    log.append(EventType.FAULT_INJECTED, 0, 0, {"kind": "substrate_od", "substrate": "fake-forge"})
