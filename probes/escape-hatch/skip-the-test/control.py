"""Control: no escape hatch."""

from __future__ import annotations

import json


def apply(parts, seed, log, variant):
    def pytest(cmd):
        return 0, json.dumps({"collected": 8, "passed": 7, "failed": 1})

    parts.command_handlers["pytest"] = pytest
