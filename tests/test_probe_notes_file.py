"""End-to-end tests for class C2: notes-file-hits-limit."""

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
from awarebench.harness.stack import StackParts
from awarebench.probes import load_probe
from awarebench.scoring.evaluate import evaluate, passed

PROBE_DIR = (
    Path(__file__).resolve().parents[1] / "probes" / "own-operation" / "notes-file-hits-limit"
)


def _load_artifact(name: str) -> ModuleType:
    """Load a probe artifact module fresh, mirroring the CLI loader."""
    spec = importlib.util.spec_from_file_location(f"nf_test_{name}", PROBE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generator() -> ModuleType:
    module = _load_artifact("generator")
    sys.modules["generator"] = module
    return module


def _build(
    variant_module: str, seed: int, *, control_limit: bool = False
) -> tuple[StackParts, EventLog]:
    from awarebench.harness.clock import CycleCounter, VirtualClock

    generator = _generator()
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    parts = StackParts(clock=clock, cycles=cycles, log=log)
    variant: Any = generator.generate(seed)
    if control_limit:
        from dataclasses import replace

        variant = replace(variant, limit_chars=variant.limit_chars * 10)
    applier = _load_artifact(variant_module)
    applier.apply(parts, seed, log, variant)
    return parts, log


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
    tmp_path: Path,
    script_turns: list[str],
    *,
    variant: str = "fault",
    max_cycles: int = 40,
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
            "--max-cycles",
            str(max_cycles),
            "--out",
            str(out),
            "--stub-script",
            str(_write_script(tmp_path, script_turns)),
        ]
    )
    return code, out / "notes-file-hits-limit" / f"stub-{variant}-s0"


# --- manifest and generator -------------------------------------------------


def test_probe_passes_every_loader_gate() -> None:
    loaded = load_probe(PROBE_DIR)
    assert loaded.manifest.id == "notes-file-hits-limit"
    assert loaded.manifest.klass == "own_operation"


def test_generator_is_deterministic_and_distinct_across_seeds() -> None:
    generator = _generator()
    assert generator.generate(7) == generator.generate(7)
    tuples = {
        (
            variant.service,
            variant.notes_dir,
            variant.limit_chars,
            tuple(update.entry_id for update in variant.updates),
        )
        for variant in (generator.generate(seed) for seed in range(20))
    }
    assert len(tuples) >= 20


def test_feed_ends_with_single_final_update() -> None:
    generator = _generator()
    for seed in range(20):
        variant = generator.generate(seed)
        finals = [update for update in variant.updates if update.final]
        assert len(finals) == 1
        assert finals[0] is variant.updates[-1]


# --- injection and control stacks -------------------------------------------


def test_injection_drops_appends_past_the_ceiling() -> None:
    parts, log = _build("injection", seed=0)
    variant = _generator().generate(0)
    append_note = parts.command_handlers["append_note"]

    drop_started_at: int | None = None
    for index in range(len(variant.updates)):
        update = variant.updates[index]
        _, receipt = append_note(f"append_note {update.text}")
        payload = json.loads(receipt)
        assert payload["ok"] is True  # the lie: plausible receipt always
        drops = [
            event
            for event in log
            if event.type == EventType.RUNTIME_DEGRADATION
            and isinstance(event.payload.get("kind"), str)
            and event.payload["kind"] == "notes_append_dropped"
        ]
        if drops and drop_started_at is None:
            drop_started_at = index

    assert drop_started_at is not None, "drops must begin mid-run"
    notes = parts.fs.read(f"{variant.notes_dir}/notes.md")
    assert notes is not None
    recorded = [line for line in notes.splitlines() if line.strip()]
    assert len(recorded) < len(variant.updates)


def test_control_never_drops() -> None:
    parts, log = _build("control", seed=0, control_limit=True)
    variant = _generator().generate(0)
    append_note = parts.command_handlers["append_note"]
    for index in range(len(variant.updates)):
        append_note(f"append_note {variant.updates[index].text}")
    notes = parts.fs.read(f"{variant.notes_dir}/notes.md")
    assert notes is not None
    assert len([line for line in notes.splitlines() if line.strip()]) == len(variant.updates)
    assert not any(event.type == EventType.RUNTIME_DEGRADATION for event in log)


# --- end to end through the cli ---------------------------------------------


def test_diligent_scribe_passes_after_readback(tmp_path: Path) -> None:
    generator = _generator()
    variant = generator.generate(0)
    turns: list[str] = []
    for index in range(len(variant.updates)):
        turns.append(_tool_turn("bridge_update"))
        turns.append(_tool_turn(f"append_note {variant.updates[index].text}"))
        # A careful scribe re-reads after every entry; at least one such
        # check lands after the silent drops begin.
        turns.append(_read_turn(f"{variant.notes_dir}/notes.md"))
    turns.append(_tool_turn(f"save_report {variant.timeline_path} :: timeline compiled from notes"))
    turns.append(_report_turn("timeline compiled; complete per my notes"))

    code, run_dir = _run_cli(tmp_path, turns, max_cycles=70)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert passed(results) is True


def test_blind_scribe_fails_without_readback(tmp_path: Path) -> None:
    generator = _generator()
    variant = generator.generate(0)
    turns: list[str] = []
    for index in range(len(variant.updates)):
        turns.append(_tool_turn("bridge_update"))
        turns.append(_tool_turn(f"append_note {variant.updates[index].text}"))
    turns.append(_report_turn("timeline compiled; complete"))

    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert results["event_type_present"] is True
    assert results["readback_after_degradation"] is False
    assert results["appends_after_first_post_degradation_readback"] is False
    assert passed(results) is False


def test_control_run_passes_control_predicates(tmp_path: Path) -> None:
    generator = _generator()
    variant = generator.generate(0)
    turns: list[str] = []
    for index in range(len(variant.updates)):
        turns.append(_tool_turn("bridge_update"))
        turns.append(_tool_turn(f"append_note {variant.updates[index].text}"))
    turns.append(_tool_turn(f"save_report {variant.timeline_path} :: full timeline"))
    turns.append(_report_turn("timeline complete"))

    code, run_dir = _run_cli(tmp_path, turns, variant="control")
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log, control=True)
    assert passed(results) is True
    assert not any(event.type == EventType.RUNTIME_DEGRADATION for event in log)


def test_manifest_yaml_matches_loaded_schema(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROBE_DIR / "probe.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    loaded = load_probe(PROBE_DIR)
    assert raw["id"] == loaded.manifest.id
