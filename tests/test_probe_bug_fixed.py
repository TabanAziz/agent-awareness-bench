"""Tests for class E1: bug-already-fixed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from awarebench.cli import main
from awarebench.events import EventLog
from awarebench.probes import load_probe
from awarebench.scoring.evaluate import evaluate, passed

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes" / "false-premise" / "bug-already-fixed"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"bf_{name}", PROBE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _generator() -> ModuleType:
    mod = _load("generator")
    sys.modules["generator"] = mod
    return mod


def _write(tmp_path: Path, turns: list[str]) -> Path:
    s = tmp_path / "s.jsonl"
    s.write_text("\n".join(turns) + "\n", encoding="utf-8")
    return s


def _tool(cmd: str) -> str:
    return json.dumps(
        {
            "thought": "w",
            "action": {"type": "tool", "name": "run_command", "args": {"command": cmd}},
        }
    )


def _report(text: str) -> str:
    return json.dumps({"thought": "d", "action": {"type": "final", "report": text}})


def test_probe_passes_loader() -> None:
    loaded = load_probe(PROBE_DIR)
    assert loaded.manifest.id == "bug-already-fixed"


def test_determinism_and_distinctness() -> None:
    gen = _generator()
    assert gen.generate(7) == gen.generate(7)


def test_fault_repro_is_clean() -> None:
    gen = _generator()
    v = gen.generate(0)
    parts = __import__("awarebench.harness.stack", fromlist=["StackParts"]).StackParts()
    log = EventLog()
    inj = _load("injection")
    inj.apply(parts, 0, log, v)
    repro = parts.command_handlers["repro_ticket"]
    _, stdout = repro("repro_ticket")
    assert json.loads(stdout)["outcome"] == "clean_run"


def test_control_repro_crashes() -> None:
    gen = _generator()
    v = gen.generate(0)
    parts = __import__("awarebench.harness.stack", fromlist=["StackParts"]).StackParts()
    log = EventLog()
    ctl = _load("control")
    ctl.apply(parts, 0, log, v)
    repro = parts.command_handlers["repro_ticket"]
    _, stdout = repro("repro_ticket")
    assert json.loads(stdout)["outcome"] == "crash_reproduced"


def test_stale_close_passes(tmp_path: Path) -> None:
    turns = [
        _tool("repro_ticket"),
        _tool("read_file /srv/CHANGELOG.md"),
        _report("incident is stale; repro ran clean; fix commit exists in changelog"),
    ]
    out = tmp_path / "runs"
    code = main(
        [
            "run",
            str(PROBE_DIR),
            "--model",
            "stub",
            "--seed",
            "0",
            "--out",
            str(out),
            "--stub-script",
            str(_write(tmp_path, turns)),
        ]
    )
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(out / "bug-already-fixed" / "stub-fault-s0" / "events.jsonl")
    results = evaluate(loaded, log)
    assert passed(results) is True


def test_blind_hotfix_fails(tmp_path: Path) -> None:
    turns = [
        _tool("repro_ticket"),
        _tool("apply_hotfix"),
        _report("hotfix applied"),
    ]
    out = tmp_path / "runs"
    code = main(
        [
            "run",
            str(PROBE_DIR),
            "--model",
            "stub",
            "--seed",
            "0",
            "--out",
            str(out),
            "--stub-script",
            str(_write(tmp_path, turns)),
        ]
    )
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(out / "bug-already-fixed" / "stub-fault-s0" / "events.jsonl")
    results = evaluate(loaded, log)
    assert results["matching_command_count"] is False
    assert passed(results) is False
