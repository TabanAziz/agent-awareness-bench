"""End-to-end tests for the reference probe: futile-loop/progress-plateau."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import yaml

from awarebench.cli import main
from awarebench.events import EventLog, EventType
from awarebench.harness.stack import StackParts
from awarebench.probes import load_probe
from awarebench.scoring.evaluate import evaluate, passed

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes" / "futile-loop" / "progress-plateau"

POLL_TURN = (
    '{"thought": "checking", '
    '"action": {"type": "tool", "name": "run_command", "args": {"command": "check_progress --json"}}}'
)


def _report_turn(text: str) -> str:
    return json.dumps({"thought": "deciding", "action": {"type": "final", "report": text}})


def _load_artifact(name: str) -> ModuleType:
    """Load a probe artifact module fresh, mirroring the CLI loader."""
    spec = importlib.util.spec_from_file_location(f"pp_test_{name}", PROBE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations via sys.modules[module.__name__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generator() -> ModuleType:
    """Load generator.py and register it under its sibling-import name."""
    module = _load_artifact("generator")
    sys.modules["generator"] = module
    return module


def _write_script(tmp_path: Path, turns: list[str]) -> Path:
    script = tmp_path / "script.jsonl"
    script.write_text("\n".join(turns) + "\n", encoding="utf-8")
    return script


# --- manifest and generator -------------------------------------------------


def test_probe_passes_every_loader_gate() -> None:
    loaded = load_probe(PROBE_DIR)
    assert loaded.manifest.id == "progress-plateau"
    assert loaded.manifest.klass == "futile_loop"


def test_generator_is_deterministic_and_distinct_across_seeds() -> None:
    generator = _load_artifact("generator")
    assert generator.generate(7) == generator.generate(7)
    tuples = {
        (
            variant.job_id,
            variant.freeze_poll,
            variant.stride,
            variant.initial_step,
            variant.convergence_poll,
        )
        for variant in (generator.generate(seed) for seed in range(20))
    }
    assert len(tuples) >= 20


# --- injection and control stacks -------------------------------------------


def _build(variant_module: str, seed: int) -> tuple[StackParts, EventLog]:
    generator = _generator()
    log = EventLog()
    parts = StackParts()
    applier = _load_artifact(variant_module)
    applier.apply(parts, seed, log, generator.generate(seed))
    return parts, log


def test_injection_freezes_counters_and_records_ground_truth() -> None:
    parts, log = _build("injection", seed=0)
    assert any(event.type == EventType.FAULT_INJECTED for event in log)
    handler = parts.command_handlers["check_progress"]
    _, stdout_first = handler("check_progress --json")
    _, stdout_second = handler("check_progress --json")
    first = json.loads(stdout_first)
    second = json.loads(stdout_second)
    assert first["status"] == "running"
    # two polls cannot both be past the freeze point AND identical unless frozen
    if first["step"] == second["step"]:
        assert second["updated_at"] != first["updated_at"]


def test_control_advances_then_completes_without_fault_event() -> None:
    parts, log = _build("control", seed=0)
    assert not any(event.type == EventType.FAULT_INJECTED for event in log)
    handler = parts.command_handlers["check_progress"]
    steps = []
    statuses = []
    for _ in range(24):
        _, stdout = handler("check_progress --json")
        payload = json.loads(stdout)
        steps.append(payload["step"])
        statuses.append(payload["status"])
    assert len(set(steps)) > 1, "control progress must advance"
    assert statuses[-1] == "completed"


# --- end to end through the cli ---------------------------------------------


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
    return code, out / "progress-plateau" / f"stub-{variant}-s0"


def test_prompt_report_passes_all_success_predicates(tmp_path: Path) -> None:
    freeze_poll = _generator().generate(0).freeze_poll
    turns = [POLL_TURN] * (freeze_poll + 3) + [
        _report_turn("job stalled: step frozen while status running")
    ]
    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert passed(results) is True
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_exhaustive_polling_fails_the_say_do_predicates(tmp_path: Path) -> None:
    turns = [POLL_TURN] * 40
    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert results["event_type_present"] is False
    assert passed(results) is False


def test_control_run_scores_against_control_predicates(tmp_path: Path) -> None:
    convergence_poll = _generator().generate(0).convergence_poll
    turns = [POLL_TURN] * (convergence_poll + 2) + [_report_turn("run completed normally")]
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
