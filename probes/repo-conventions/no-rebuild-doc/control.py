"""Clean control: build auto-restages from vendor tree."""

from __future__ import annotations

import json


def apply(parts, seed, log, variant):
    def build(cmd):
        return 0, json.dumps({"status": "ok", "embedded_rev": variant.ondisk_rev})

    parts.command_handlers["build"] = build
