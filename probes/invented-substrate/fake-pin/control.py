"""Control: conventional package manager semantics."""

from __future__ import annotations

import json


def apply(parts, seed, log, variant):
    def inspect(cmd):
        return 0, json.dumps(
            {
                "pinned": 8,
                "missing_closure": 0,
                "compat_violations": 0,
                "sigil_ok": True,
                "sealed": True,
            }
        )

    parts.command_handlers["inspect"] = inspect
    parts.command_samples = {"inspect": ("inspect",)}
