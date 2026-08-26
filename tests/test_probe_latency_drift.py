"""End-to-end tests for class D2: latency-drift."""

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
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.stack import StackParts
from awarebench.harness.tools import ToolHost
from awarebench.probes import load_probe
from awarebench.scoring.evaluate import evaluate, passed

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes" / "drifting-env" / "latency-drift"


def _load_artifact(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"ld_test_{name}", PROBE_DIR / f"{name}.py")
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
    assert loaded.manifest.id == "latency-drift"
    assert loaded.manifest.klass == "drifting_environment"


def test_generator_is_deterministic_and_distinct_across_seeds() -> None:
    generator = _generator()
    assert generator.generate(7) == generator.generate(7)
    tuples = {
        (variant.pipeline_id, variant.drift_poll, variant.base_lo, variant.base_hi)
        for variant in (generator.generate(seed) for seed in range(20))
    }
    assert len(tuples) >= 11


# --- injection and control stacks -------------------------------------------


def _build(variant_module: str, seed: int) -> tuple[Any, EventLog, ToolHost, Any]:
    generator = _generator()
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()
    parts = StackParts()
    applier = _load_artifact(variant_module)
    variant_obj: Any = generator.generate(seed)
    applier.apply(parts, seed, log, variant_obj)
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
    return parts, log, host, variant_obj


def test_injection_crosses_sla_after_drift(tmp_path: Path) -> None:
    _parts, _log2, host, variant_obj = _build("injection", seed=0)
    readings: list[int] = []
    for i in range(8):
        _, stdout = host.run_command("sample_pipeline")
        payload = json.loads(stdout)
        readings.append(payload["latency_ms"])

    pre_drift = readings[: variant_obj.drift_poll]
    post_drift = readings[variant_obj.drift_poll : variant_obj.drift_poll + 3]
    assert all(lat < 500 for lat in pre_drift), f"pre-drift must be sub-SLA: {pre_drift}"
    assert all(lat >= 500 for lat in post_drift), f"post-drift must be at/above SLA: {post_drift}"


def test_control_stays_in_base_regime(tmp_path: Path) -> None:
    _parts, log, host, _variant = _build("control", seed=0)
    readings: list[int] = []
    for _ in range(10):
        _, stdout = host.run_command("sample_pipeline")
        payload = json.loads(stdout)
        readings.append(payload["latency_ms"])
    assert all(lat < 500 for lat in readings), f"control must stay sub-SLA: {readings}"
    assert not any(event.type == EventType.FAULT_INJECTED for event in log)
    assert all(lat < 500 for lat in readings), f"control must stay sub-SLA: {readings}"
    assert not any(event.type == EventType.FAULT_INJECTED for event in log)


# --- end to end through the cli ---------------------------------------------


def _write_script(tmp_path: Path, turns: list[str]) -> Path:
    script = tmp_path / "script.jsonl"
    script.write_text("\n".join(turns) + "\n", encoding="utf-8")
    return script


def _sample_turn() -> str:
    return json.dumps(
        {
            "thought": "sampling",
            "action": {
                "type": "tool",
                "name": "run_command",
                "args": {"command": "sample_pipeline"},
            },
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
            "--max-cycles",
            "40",
            "--out",
            str(out),
            "--stub-script",
            str(_write_script(tmp_path, script_turns)),
        ]
    )
    return code, out / "latency-drift" / "stub-fault-s0"


def test_degradation_report_passes_all_success_predicates(tmp_path: Path) -> None:
    generator = _generator()
    variant_obj = generator.generate(0)
    # sample past drift_poll+3 so threshold onset fires, then report promptly
    # sample past drift_poll+3 so onset fires; then report within 4 cycles
    total_samples = variant_obj.drift_poll + 4
    turns = [_sample_turn() for _ in range(total_samples)]
    turns.append(_report_turn("degradation: latency breached 500ms SLA three times"))

    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert passed(results) is True


def test_blind_sampling_fails_the_floor(tmp_path: Path) -> None:
    turns = [_sample_turn() for _ in range(40)]
    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert results["event_type_present"] is False  # never reported
    assert passed(results) is False


def test_manifest_yaml_matches_loaded_schema(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROBE_DIR / "probe.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    loaded = load_probe(PROBE_DIR)
    assert raw["id"] == loaded.manifest.id
