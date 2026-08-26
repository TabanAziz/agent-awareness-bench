"""Executable contract for the no-rebuild-doc remediation path."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from awarebench.events import EventLog
from awarebench.harness.stack import StackParts

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes" / "repo-conventions" / "no-rebuild-doc"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"no_rebuild_test_{name}", PROBE_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fault_can_be_remediated_by_restage_then_build() -> None:
    generator = _load("generator")
    sys.modules["generator"] = generator
    injection = _load("injection")
    variant = generator.generate(0)
    parts = StackParts()
    injection.apply(parts, 0, EventLog(), variant)

    build = parts.command_handlers["build"]
    restage = parts.command_handlers["restage"]
    first = json.loads(build("build")[1])
    staged = json.loads(restage("restage")[1])
    rebuilt = json.loads(build("build")[1])

    assert first["embedded_rev"] == variant.staged_rev
    assert staged["staged_rev"] == variant.ondisk_rev
    assert rebuilt["embedded_rev"] == variant.ondisk_rev
    assert rebuilt["embedded_rev"] >= variant.floor
