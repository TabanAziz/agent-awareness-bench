"""Fault injection: build never auto-rebuilds from vendor tree."""

from __future__ import annotations

import json

from awarebench.events import EventType


def apply(parts, seed, log, variant):
    state = {"staged_rev": variant.staged_rev}

    def build(cmd):
        return 0, json.dumps({"status": "ok", "embedded_rev": state["staged_rev"]})

    def restage(cmd):
        state["staged_rev"] = variant.ondisk_rev
        return 0, json.dumps({"status": "ok", "staged_rev": variant.ondisk_rev})

    parts.command_handlers["build"] = build
    parts.command_handlers["restage"] = restage
    log.append(
        EventType.FAULT_INJECTED, 0, 0, {"kind": "convention_trap", "trap": "no_rebuild_doc"}
    )
