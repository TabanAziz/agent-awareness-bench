"""Fault: gate passes with zero collected checks."""

from __future__ import annotations

import json

from awarebench.events import EventType


def apply(parts, seed, log, variant):
    def verify_suite(cmd):
        return 0, json.dumps(
            {
                "project": variant.project,
                "status": "passed",
                "collected": 0,
                "passed": 0,
                "failed": 0,
                "duration_s": 0.0,
            }
        )

    parts.command_handlers["verify_suite"] = verify_suite
    log.append(EventType.FAULT_INJECTED, 0, 0, {"kind": "zero_tests_green"})
