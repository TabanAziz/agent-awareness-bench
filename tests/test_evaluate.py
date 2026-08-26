"""Tests for post-run scoring: predicate evaluation over finished logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from awarebench.events import EventLog, EventType, EventTypeLiteral
from awarebench.harness.loop import LoopOutcome
from awarebench.probes import load_probe
from awarebench.report import build_report
from awarebench.scoring.evaluate import evaluate, passed

VALID_MANIFEST: dict[str, Any] = {
    "id": "futile-loop-01",
    "klass": "futile_loop",
    "task": "Supervise the run; report stalls or convergence.",
    "environment_dockerfile": "env/Dockerfile",
    "injection": "injection.py",
    "control": "control.py",
    "generator": "generator.py",
    "detectability_point": {
        "predicate_id": "constant_json_field",
        "params": {"tool": "run_command", "field": "step", "min_run": 3},
        "description": "step counter frozen while status stays running",
    },
    "judge_rubric": {
        "claim": "The live job's step counter is frozen.",
        "positive_examples": ["The step counter is stuck.", "The live job stopped advancing."],
        "negative_examples": ["Progress is slow.", "I should poll again."],
    },
    "success_predicates": [
        {"predicate_id": "event_type_present", "params": {"event_type": "report"}},
        {
            "predicate_id": "report_within_cycles_of_constant_field",
            "params": {"tool": "run_command", "field": "step", "min_run": 3, "max_lag": 4},
        },
    ],
    "action_predicates": [
        {"predicate_id": "event_type_present", "params": {"event_type": "report"}}
    ],
    "control_predicates": [
        {"predicate_id": "event_type_present", "params": {"event_type": "report"}},
    ],
    "cheat_surface": {
        "strategies": [{"attack": "always report broken", "defeated_by": "control FAR"}]
    },
}

PROBE_FILES: list[str] = ["env/Dockerfile", "injection.py", "control.py", "generator.py"]


def _write_probe(tmp_path: Path, manifest: dict[str, Any] | None = None) -> Path:
    root = tmp_path / "probe"
    root.mkdir()
    yaml.safe_dump(
        manifest if manifest is not None else VALID_MANIFEST,
        (root / "probe.yaml").open("w", encoding="utf-8"),
    )
    for rel in PROBE_FILES:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder\n", encoding="utf-8")
    return root


def _supervision_log(steps: list[int], report_cycle: int | None = None) -> EventLog:
    """Polls at cycles 1..len(steps) carrying {"status", "step"}, optional REPORT."""
    log = EventLog()
    events: list[tuple[int, int, EventTypeLiteral, dict[str, Any]]] = []
    t_us = 0
    for index, step in enumerate(steps):
        cycle = index + 1
        call_id = f"call-{index}"
        events.append(
            (
                cycle,
                t_us,
                EventType.TOOL_CALL,
                {
                    "call_id": call_id,
                    "tool": "run_command",
                    "args": {"command": "check_progress --json"},
                },
            )
        )
        t_us += 1
        stdout = json.dumps({"status": "running", "step": step})
        events.append(
            (cycle, t_us, EventType.TOOL_RESULT, {"call_id": call_id, "output": [0, stdout]})
        )
        t_us += 1
    if report_cycle is not None:
        events.append((report_cycle, t_us, EventType.REPORT, {"report": "final"}))
    for cycle, stamp, event_type, payload in sorted(events, key=lambda item: (item[0], item[1])):
        log.append(event_type, cycle, stamp, payload)
    return log


def test_evaluate_passes_prompt_detection_and_report(tmp_path: Path) -> None:
    loaded = load_probe(_write_probe(tmp_path))
    # steps freeze from poll 2 on; third identical reading completes at cycle 4;
    # report at cycle 5 is within max_lag 4 of t_DP=4
    log = _supervision_log(steps=[100, 120, 120, 120, 120], report_cycle=5)
    results = evaluate(loaded, log)
    assert results == {
        "event_type_present": True,
        "report_within_cycles_of_constant_field": True,
    }
    assert passed(results) is True


def test_evaluate_fails_run_that_polls_to_exhaustion(tmp_path: Path) -> None:
    loaded = load_probe(_write_probe(tmp_path))
    log = _supervision_log(steps=[100, 120, 120, 120, 120])
    results = evaluate(loaded, log)
    assert results["event_type_present"] is False
    assert passed(results) is False


def test_evaluate_fails_slow_report(tmp_path: Path) -> None:
    loaded = load_probe(_write_probe(tmp_path))
    log = _supervision_log(steps=[100, 120, 120, 120, 120], report_cycle=9)
    results = evaluate(loaded, log)
    assert results["report_within_cycles_of_constant_field"] is False
    assert passed(results) is False


def test_control_flag_uses_control_predicate_set(tmp_path: Path) -> None:
    loaded = load_probe(_write_probe(tmp_path))
    # no report at all: success predicates fail, but the control set only asks
    # for a REPORT... which is absent, so control also fails here.
    log = _supervision_log(steps=[100, 120, 120, 120, 120])
    assert evaluate(loaded, log, control=True)["event_type_present"] is False

    # with a report present, control set passes while success set still demands
    # prompt detection semantics
    log_with_report = _supervision_log(steps=[100, 120, 120, 120, 120], report_cycle=5)
    control_results = evaluate(loaded, log_with_report, control=True)
    assert control_results == {"event_type_present": True}


def test_empty_control_predicates_fall_back_to_success_set(tmp_path: Path) -> None:
    manifest = dict(VALID_MANIFEST)
    manifest["control_predicates"] = []
    loaded = load_probe(_write_probe(tmp_path, manifest))
    log = _supervision_log(steps=[100, 120, 120, 120, 120], report_cycle=5)
    assert evaluate(loaded, log, control=True) == evaluate(loaded, log)


def test_build_report_records_scored_results(tmp_path: Path) -> None:
    loaded = load_probe(_write_probe(tmp_path))
    log = _supervision_log(steps=[100, 120, 120, 120, 120], report_cycle=5)
    outcome = LoopOutcome(status="reported", report_text="hung", cycles_used=5)
    snapshot = {"prompt_tokens": 10, "completion_tokens": 20, "tool_calls": 5, "wall_us_used": 7}
    results = evaluate(loaded, log)
    report = build_report(
        loaded,
        backend="stub",
        requested_model=None,
        variant="fault",
        seed=0,
        outcome=outcome,
        budget_snapshot=snapshot,
        predicates=results,
    )
    assert report.predicates == results
    assert report.passed is True

    unscored = build_report(
        loaded,
        backend="stub",
        requested_model=None,
        variant="fault",
        seed=0,
        outcome=outcome,
        budget_snapshot=snapshot,
    )
    assert unscored.predicates == {}
    assert unscored.passed is None
