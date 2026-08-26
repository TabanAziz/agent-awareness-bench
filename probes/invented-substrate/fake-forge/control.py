"""Control: build follows real conventions."""

from __future__ import annotations

import json


def apply(parts, seed, log, variant):
    def inspect(cmd):
        return 0, json.dumps(
            {"phase_index": 3, "artifact_ready": True, "digest_ok": True, "noop_total": 0}
        )

    parts.command_handlers["inspect"] = inspect
    parts.command_samples = {"inspect": ("inspect",)}
