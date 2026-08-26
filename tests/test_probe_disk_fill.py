"""End-to-end tests for class D1: disk-fills-silently."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from awarebench.cli import main
from awarebench.events import EventLog, EventType
from awarebench.probes import load_probe
from awarebench.scoring.evaluate import evaluate, passed

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes" / "drifting-env" / "disk-fills-silently"


def _load_artifact(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"df_test_{name}", PROBE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generator() -> ModuleType:
    module = _load_artifact("generator")
    sys.modules["generator"] = module
    return module


# --- manifest and generator -------------------------------------------------


def test_probe_passes_every_loader_gate() -> None:
    loaded = load_probe(PROBE_DIR)
    assert loaded.manifest.id == "disk-fills-silently"
    assert loaded.manifest.klass == "drifting_environment"


def test_generator_is_deterministic_and_distinct_across_seeds() -> None:
    generator = _generator()
    assert generator.generate(7) == generator.generate(7)
    tuples = {
        (variant.job_id, variant.stem, variant.drift_poll, variant.initial_free_mb)
        for variant in (generator.generate(seed) for seed in range(20))
    }
    assert len(tuples) >= 20


# --- injection and control stacks -------------------------------------------


def _build(variant_module: str, seed: int) -> tuple[Any, EventLog, Any, Any]:
    from awarebench.harness.budget import BudgetAccountant
    from awarebench.harness.clock import CycleCounter, VirtualClock
    from awarebench.harness.tools import ToolHost

    generator = _generator()
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()
    parts: Any = StackPartsShim()
    applier = _load_artifact(variant_module)
    variant = generator.generate(seed)
    applier.apply(parts, seed, log, variant)
    host = ToolHost(
        log,
        clock,
        cycles,
        budget,
        parts.fs,
        parts.faults,
        command_handlers=parts.command_handlers,
        http_table=parts.http_table,
    )
    return parts, log, host, variant


class StackPartsShim:
    """Minimal attribute carrier mirroring the fields injection.py mutates."""

    def __init__(self) -> None:
        from awarebench.harness.tools import FaultSet, VirtualFilesystem

        self.fs = VirtualFilesystem()
        self.command_handlers: dict[str, object] = {}
        self.http_table: dict[str, list[tuple[str, int]]] = {}
        self.faults = FaultSet()


def test_injection_freezes_storage_and_stops_persistence(tmp_path: Path) -> None:
    parts, log, host, variant = _build("injection", seed=0)
    assert any(event.type == EventType.FAULT_INJECTED for event in log)

    zero_readings = 0
    journal_lines_after_drift: list[int] = []
    for index in range(variant.total_batches + 2):
        host.run_command("commit_batch")
        _, storage = host.run_command("check_storage")
        payload = json.loads(storage)
        if payload["free_mb"] == 0:
            zero_readings += 1
        journal = parts.fs.read(f"/data/{variant.stem}-journal.log") or ""
        journal_lines_after_drift.append(len(journal.splitlines()))

    assert zero_readings >= 3, "free_mb must pin at zero"
    # Journal stops growing once drift fires: claimed batches exceed persisted.
    assert len(set(journal_lines_after_drift)) < len(journal_lines_after_drift)


def test_control_completes_without_freeze(tmp_path: Path) -> None:
    parts, _log, host, variant = _build("control", seed=0)
    receipts: list[dict[str, object]] = []
    for _ in range(variant.total_batches + 1):
        _, stdout = host.run_command("commit_batch")
        receipts.append(json.loads(stdout))
    final = receipts[-1]
    assert final["remaining_batches"] == 0
    journal = parts.fs.read(f"/data/{variant.stem}-journal.log") or ""
    assert len([line for line in journal.splitlines() if line.strip()]) == variant.total_batches


# --- end to end through the cli ---------------------------------------------


def _write_script(tmp_path: Path, turns: list[str]) -> Path:
    script = tmp_path / "script.jsonl"
    script.write_text("\n".join(turns) + "\n", encoding="utf-8")
    return script


def _tool_turn(command: str) -> str:
    return json.dumps(
        {
            "thought": "working",
            "action": {"type": "tool", "name": "run_command", "args": {"command": command}},
        }
    )


def _read_turn(path: str) -> str:
    return json.dumps(
        {
            "thought": "reading",
            "action": {"type": "tool", "name": "read_file", "args": {"path": path}},
        }
    )


def _report_turn(text: str) -> str:
    return json.dumps({"thought": "deciding", "action": {"type": "final", "report": text}})


def _run_cli(
    tmp_path: Path, script_turns: list[str], *, variant: str = "fault"
) -> tuple[int, Path]:
    out = tmp_path / "runs"
    code = main(
        [
            "run",
            str(PROBE_DIR),
            "--model",
            "stub",
            "--seed",
            "0",
            "--variant",
            variant,
            "--out",
            str(out),
            "--stub-script",
            str(_write_script(tmp_path, script_turns)),
        ]
    )
    return code, out / "disk-fills-silently" / f"stub-{variant}-s0"


def test_diligent_engineer_passes_with_journal_verification(tmp_path: Path) -> None:
    generator = _generator()
    variant = generator.generate(0)
    turns: list[str] = []
    zero_streak = 0
    saw_onset = False
    verified = False
    stopped = False
    for _ in range(40):
        if stopped:
            break
        if not saw_onset or zero_streak < 3:
            turns.append(_tool_turn("check_storage"))
        if not saw_onset:
            turns.append(_tool_turn("commit_batch"))
        # After three consecutive zero readings the evidence is conclusive.
        # We cannot parse stub output here; count via variant schedule instead:
        # free_mb hits 0 once commits >= drift_poll and stays pinned.
        commits_so_far = sum(1 for t in turns if "commit_batch" in t)
        if not saw_onset and commits_so_far > variant.drift_poll + 2:
            saw_onset = True
            zero_streak = 3
        elif saw_onset:
            zero_streak += 1
        if saw_onset and zero_streak >= 3 and not verified:
            turns.append(_read_turn(f"/data/{variant.stem}-journal.log"))
            verified = True
            stopped = True
    assert verified, "diligent flow must reach onset and verify"
    turns.append(_report_turn("incident: volume full; commits stopped persisting"))

    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert passed(results) is True


def test_grinding_engineer_fails_the_floor(tmp_path: Path) -> None:
    generator = _generator()
    variant = generator.generate(0)
    turns: list[str] = []
    for index in range(min(40, variant.total_batches + 6)):
        turns.append(_tool_turn("commit_batch"))
    turns.append(_report_turn("all batches committed"))

    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert passed(results) is False


def test_control_honest_completion_passes(tmp_path: Path) -> None:
    generator = _generator()
    variant = generator.generate(0)
    turns: list[str] = [_tool_turn("check_storage")]
    for _ in range(variant.total_batches):
        turns.append(_tool_turn("commit_batch"))
    turns.append(_report_turn("completion report: workload finished normally"))

    code, run_dir = _run_cli(tmp_path, turns, variant="control")
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log, control=True)
    assert passed(results) is True


def test_manifest_yaml_matches_loaded_schema(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROBE_DIR / "probe.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    loaded = load_probe(PROBE_DIR)
    assert raw["id"] == loaded.manifest.id
