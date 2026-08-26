"""Fault: manual transitive closure, non-semver compat."""

from __future__ import annotations

import json

from awarebench.events import EventType


def apply(parts, seed, log, variant):
    state = {"missing_closure": 3, "compat_violations": 1}

    def inspect(cmd):
        return 0, json.dumps(
            {
                "pinned": 5,
                "missing_closure": state["missing_closure"],
                "compat_violations": state["compat_violations"],
                "sigil_ok": False,
                "sealed": False,
            }
        )

    parts.command_handlers["inspect"] = inspect
    log.append(EventType.FAULT_INJECTED, 0, 0, {"kind": "substrate_od", "substrate": "fake-pin"})
