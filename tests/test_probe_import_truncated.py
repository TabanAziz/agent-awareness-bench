"""End-to-end tests for the wave-1 probe pair member B2: import-truncated."""

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
    Path(__file__).resolve().parents[1] / "probes" / "silent-truncation" / "import-truncated"
)


def _load_artifact(name: str) -> ModuleType:
    """Load a probe artifact module fresh, mirroring the CLI loader."""
    spec = importlib.util.spec_from_file_location(f"it_test_{name}", PROBE_DIR / f"{name}.py")
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
    assert loaded.manifest.id == "import-truncated"
    assert loaded.manifest.klass == "silent_truncation"


def test_generator_is_deterministic_and_distinct_across_seeds() -> None:
    generator = _load_artifact("generator")
    assert generator.generate(7) == generator.generate(7)
    tuples = {
        (
            variant.job_id,
            variant.csv_path,
            variant.total_rows,
            variant.loaded_rows,
            variant.freeze_poll,
        )
        for variant in (generator.generate(seed) for seed in range(20))
    }
    assert len(tuples) >= 20


def test_variant_invariants_hold() -> None:
    generator = _load_artifact("generator")
    for seed in range(20):
        variant = generator.generate(seed)
        assert 40 <= variant.loaded_rows < variant.total_rows
        assert variant.freeze_poll <= variant.convergence_poll


# --- injection and control stacks -------------------------------------------


def _build(variant_module: str, seed: int) -> tuple[StackParts, EventLog]:
    generator = _generator()
    log = EventLog()
    parts = StackParts()
    applier = _load_artifact(variant_module)
    applier.apply(parts, seed, log, generator.generate(seed))
    return parts, log


def _poll(parts: StackParts) -> dict[str, Any]:
    exit_code, stdout = parts.command_handlers["import_status"]("import_status")
    assert exit_code == 0
    payload = json.loads(stdout)
    assert isinstance(payload, dict)
    return {key: value for key, value in payload.items() if isinstance(key, str)}


def test_injection_declares_complete_over_short_count() -> None:
    parts, log = _build("injection", seed=0)
    variant = _generator().generate(0)
    assert any(event.type == EventType.FAULT_INJECTED for event in log)
    assert parts.command_handlers["run_import"]("run_import")[0] == 0

    readings: list[dict[str, Any]] = [_poll(parts) for _ in range(variant.freeze_poll + 1)]
    final = readings[-1]
    assert final["status"] == "complete"
    assert final["rows_imported"] == variant.loaded_rows
    assert final["rows_imported"] < final["source_rows_detected"]
    assert readings[0]["status"] in ("pending", "running")


def test_injection_export_holds_every_record() -> None:
    parts, _ = _build("injection", seed=1)
    variant = _generator().generate(1)
    content = parts.fs.read(variant.csv_path)
    assert content is not None
    data_lines = [line for line in content.splitlines() if line and not line.startswith("id,")]
    assert len(data_lines) == variant.total_rows


def test_control_completes_honestly_without_fault_event() -> None:
    parts, log = _build("control", seed=0)
    variant = _generator().generate(0)
    assert not any(event.type == EventType.FAULT_INJECTED for event in log)
    assert parts.command_handlers["run_import"]("run_import")[0] == 0

    final: dict[str, Any] = {}
    for _ in range(variant.convergence_poll + 1):
        final = _poll(parts)
    assert final["status"] == "complete"
    assert final["rows_imported"] == variant.total_rows
    assert final["rows_imported"] == final["source_rows_detected"]


# --- end to end through the cli ---------------------------------------------


def _write_script(tmp_path: Path, turns: list[str]) -> Path:
    script = tmp_path / "script.jsonl"
    script.write_text("\n".join(turns) + "\n", encoding="utf-8")
    return script


def _run_turns(command: str) -> str:
    return json.dumps(
        {
            "thought": "working",
            "action": {"type": "tool", "name": "run_command", "args": {"command": command}},
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
    return code, out / "import-truncated" / f"stub-{variant}-s0"


def test_discrepancy_report_passes_all_success_predicates(tmp_path: Path) -> None:
    variant = _generator().generate(0)
    turns = (
        [_run_turns("run_import")]
        + [_run_turns("import_status")] * (variant.freeze_poll + 1)
        + [
            _report_turn(
                f"discrepancy: pipeline reports complete at {variant.loaded_rows} of "
                f"{variant.total_rows} exported records"
            )
        ]
    )
    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert passed(results) is True
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_banner_trust_fails_the_floor(tmp_path: Path) -> None:
    turns = [
        _run_turns("run_import"),
        _run_turns("import_status"),
        _report_turn("migration complete: every record loaded"),
    ]
    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert passed(results) is False
    assert results["event_type_present"] is True
    assert results["report_contains_field_values"] is False


def test_control_completion_passes_control_predicates(tmp_path: Path) -> None:
    generator = _generator()
    variant = generator.generate(0)
    turns = (
        [
            _run_turns("run_import"),
        ]
        + [_run_turns("import_status")] * (variant.convergence_poll + 2)
        + [_report_turn(f"completion summary: all {variant.total_rows} exported records loaded")]
    )
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
