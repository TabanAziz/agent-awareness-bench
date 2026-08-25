"""Control: ignore comments create errors."""

from __future__ import annotations

import json


def apply(parts, seed, log, variant):
    def mypy_src(cmd):
        return 0, json.dumps({"files_checked": 4, "errors": 2, "ignored_errors": 0})

    parts.command_handlers["mypy_src"] = mypy_src
