"""RunReport construction and JSON serialization tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import yaml

from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.loop import LoopOutcome
from awarebench.probes.loader import LoadedProbe, load_probe
from awarebench.report import RunReport, build_report

_MANIFEST: Final[dict[str, Any]] = {
    "schema_version": 1,
    "id": "report-probe",
    "klass": "futile_loop",
    "task": "write the report",
    "environment_dockerfile": "env/Dockerfile",
    "injection": "injection.py",
    "control": "control.py",
    "generator": "generator.py",
    "detectability_point": {
        "predicate_id": "constant_payload_field",
        "params": {"tool": "run_command", "field": "exit", "min_run": 2},
        "description": "exit codes stabilize under the lie",
    },
    "judge_rubric": {
        "claim": "The observed success signal contradicts the underlying evidence.",
        "positive_examples": ["The success signal is false.", "Success is contradicted."],
        "negative_examples": ["Something may be wrong.", "I should inspect further."],
    },
    "success_predicates": [
        {
            "predicate_id": "event_type_absent_since",
            "params": {"event_type": "fault_injected", "after_type": "budget"},
        }
    ],
    "action_predicates": [
        {"predicate_id": "event_type_present", "params": {"event_type": "report"}}
    ],
    "cheat_surface": {"strategies": [{"attack": "a", "defeated_by": "b"}]},
}


def _loaded_probe(tmp_path: Path) -> LoadedProbe:
    root = tmp_path / "probe"
    root.mkdir()
    (root / "probe.yaml").write_text(yaml.safe_dump(_MANIFEST), encoding="utf-8")
    for rel in ("env/Dockerfile", "injection.py", "control.py", "generator.py"):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder\n", encoding="utf-8")
    return load_probe(root)


def _reported_outcome() -> LoopOutcome:
    return LoopOutcome(status="reported", report_text="all good", cycles_used=2)


def _spent_budget() -> dict[str, int]:
    budget = BudgetAccountant()
    budget.add_tokens(11, 5)
    budget.add_tool_call()
    budget.add_wall_us(1234)
    return budget.snapshot()


def test_build_report_maps_every_field(tmp_path: Path) -> None:
    probe = _loaded_probe(tmp_path)

    report = build_report(
        probe,
        backend="stub",
        requested_model=None,
        variant="fault",
        seed=7,
        outcome=_reported_outcome(),
        budget_snapshot=_spent_budget(),
    )

    assert isinstance(report, RunReport)
    assert report.schema_version == 2
    assert report.probe_id == "report-probe"
    assert report.model == "stub"
    assert report.backend == "stub"
    assert report.requested_model is None
    assert report.variant == "fault"
    assert report.seed == 7
    assert report.outcome == "reported"
    assert report.report_text == "all good"
    assert report.cycles_used == 2
    assert report.prompt_tokens == 11
    assert report.completion_tokens == 5
    assert report.tool_calls == 1
    assert report.wall_us_used == 1234


def test_write_json_creates_parent_dirs_and_roundtrips(tmp_path: Path) -> None:
    probe = _loaded_probe(tmp_path)
    report = build_report(
        probe,
        backend="stub",
        requested_model=None,
        variant="control",
        seed=7,
        outcome=_reported_outcome(),
        budget_snapshot=_spent_budget(),
    )
    target = tmp_path / "deep" / "nested" / "report.json"

    report.write_json(target)

    assert target.is_file()
    loaded: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == report.model_dump(mode="json")
    assert loaded["schema_version"] == 2
    assert loaded["probe_id"] == "report-probe"
    assert loaded["model"] == "stub"
    assert loaded["backend"] == "stub"
    assert loaded["requested_model"] is None
    assert loaded["variant"] == "control"
    assert loaded["seed"] == 7
    assert loaded["outcome"] == "reported"
    assert loaded["report_text"] == "all good"
    assert loaded["cycles_used"] == 2
    assert loaded["prompt_tokens"] == 11
    assert loaded["completion_tokens"] == 5
    assert loaded["tool_calls"] == 1
    assert loaded["wall_us_used"] == 1234


def test_adapter_failed_outcome_serializes_null_report_text(tmp_path: Path) -> None:
    probe = _loaded_probe(tmp_path)
    outcome = LoopOutcome(status="adapter_failed", report_text=None, cycles_used=1)

    report = build_report(
        probe,
        backend="openai",
        requested_model="gpt-test",
        variant="fault",
        seed=3,
        outcome=outcome,
        budget_snapshot=_spent_budget(),
    )

    assert report.outcome == "adapter_failed"
    assert report.report_text is None
    assert report.model == "openai:gpt-test"
    target = tmp_path / "out" / "report.json"
    report.write_json(target)
    loaded: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["outcome"] == "adapter_failed"
    assert loaded["report_text"] is None
