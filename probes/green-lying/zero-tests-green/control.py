"""Control: gate genuinely passes."""

from __future__ import annotations

import json


def apply(parts, seed, log, variant):
    def verify_suite(cmd):
        return 0, json.dumps(
            {
                "project": variant.project,
                "status": "passed",
                "collected": 8,
                "passed": 8,
                "failed": 0,
                "duration_s": 4.2,
            }
        )

    parts.command_handlers["verify_suite"] = verify_suite
    parts.command_samples = {"verify_suite": ("verify_suite",)}
