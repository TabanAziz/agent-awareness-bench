"""Aggregate semantic metrics reject unavailable inputs instead of imputing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from awarebench.events import EventLog, EventType
from awarebench.probes import load_probe
from awarebench.scoring import metrics
from awarebench.scoring.metrics import RunMetricInput, aggregate_metrics


def test_zero_of_two_detection_produces_zero_detection_and_zero_action_gap() -> None:
    summary = aggregate_metrics(
        [
            RunMetricInput(detected=False, acted_within_k=False, disagreements=0, excerpts=3),
            RunMetricInput(detected=False, acted_within_k=False, disagreements=0, excerpts=4),
        ]
    )

    assert summary.n == 2
    assert summary.detection_rate == 0.0
    assert summary.action_gap == 0.0
    assert summary.judge_disagreement_rate == 0.0


@pytest.mark.parametrize(
    ("detected", "acted", "latency", "expected_gap"),
    [(False, False, None, 0.0), (True, True, 2, 0.0), (True, False, 4, 1.0)],
)
def test_single_run_action_gap_cases(
    detected: bool,
    acted: bool,
    latency: int | None,
    expected_gap: float,
) -> None:
    summary = aggregate_metrics(
        [
            RunMetricInput(
                detected=detected,
                acted_within_k=acted,
                detection_latency=latency,
                disagreements=0,
                excerpts=1,
            )
        ]
    )
    assert summary.action_gap == expected_gap
    assert summary.median_detection_latency == (float(latency) if latency is not None else None)


def test_control_false_alarm_is_far_one_but_action_gap_zero() -> None:
    summary = aggregate_metrics(
        [
            RunMetricInput(
                detected=False,
                acted_within_k=False,
                false_alarm=True,
                disagreements=0,
                excerpts=1,
            )
        ],
        control=True,
    )

    assert summary.false_alarm_rate == 1.0
    assert summary.detection_rate == 0.0
    assert summary.action_gap == 0.0


def test_control_far_is_unavailable_when_no_false_alarm_evidence_was_supplied() -> None:
    summary = aggregate_metrics(
        [RunMetricInput(detected=False, acted_within_k=False, disagreements=0, excerpts=1)],
        control=True,
    )

    assert summary.false_alarm_rate is None


def _progress_control_log(losses: list[float], report_cycle: int) -> EventLog:
    log = EventLog()
    for cycle, loss in enumerate(losses):
        call_id = f"progress-{cycle}"
        log.append(
            EventType.TOOL_CALL,
            cycle,
            cycle * 10,
            {
                "call_id": call_id,
                "tool": "run_command",
                "args": {"command": "check_progress"},
            },
        )
        log.append(
            EventType.TOOL_RESULT,
            cycle,
            cycle * 10 + 1,
            {
                "call_id": call_id,
                "tool": "run_command",
                "output": [0, json.dumps({"eval_loss": loss})],
            },
        )
    log.append(EventType.REPORT, report_cycle, report_cycle * 10, {"report": "finished"})
    return log


def test_false_alarm_is_derived_from_explicit_control_event_predicate() -> None:
    loaded = load_probe(Path("probes/futile-loop/progress-plateau"))

    early = metrics.derive_false_alarm(
        loaded,
        _progress_control_log([0.7], 1),
        control=True,
    )
    after_convergence = metrics.derive_false_alarm(
        loaded,
        _progress_control_log([0.3, 0.2], 2),
        control=True,
    )

    assert early is True
    assert after_convergence is False


def test_false_alarm_is_unavailable_without_explicit_control_predicate() -> None:
    loaded = load_probe(Path("probes/abstract/frozen-dashboard"))

    assert metrics.derive_false_alarm(loaded, EventLog(), control=True) is None


def test_false_alarm_is_unavailable_for_fault_arm_even_with_matching_events() -> None:
    loaded = load_probe(Path("probes/futile-loop/progress-plateau"))

    assert (
        metrics.derive_false_alarm(
            loaded,
            _progress_control_log([0.7], 1),
            control=False,
        )
        is None
    )


def test_unresolved_semantic_input_makes_metrics_unavailable() -> None:
    summary = aggregate_metrics(
        [RunMetricInput(detected=None, acted_within_k=None, disagreements=1, excerpts=2)]
    )

    assert summary.detection_rate is None
    assert summary.action_gap is None
    assert summary.judge_disagreement_rate == 0.5


def test_unscorable_action_preserves_detection_but_makes_ag_unavailable() -> None:
    summary = aggregate_metrics(
        [
            RunMetricInput(
                detected=True,
                acted_within_k=None,
                detection_latency=2,
                disagreements=0,
                excerpts=1,
            )
        ]
    )

    assert summary.detection_rate == 1.0
    assert summary.median_detection_latency == 2.0
    assert summary.action_gap is None


def test_acted_cannot_be_true_without_detection() -> None:
    with pytest.raises(ValueError, match="without detection"):
        RunMetricInput(detected=False, acted_within_k=True, disagreements=0, excerpts=1)


def test_detection_requires_latency_and_reports_median_over_detected_runs() -> None:
    with pytest.raises(ValueError, match="latency"):
        RunMetricInput(detected=True, acted_within_k=False, disagreements=0, excerpts=1)

    summary = aggregate_metrics(
        [
            RunMetricInput(
                detected=True,
                acted_within_k=True,
                detection_latency=1,
                disagreements=0,
                excerpts=1,
            ),
            RunMetricInput(
                detected=False,
                acted_within_k=False,
                disagreements=0,
                excerpts=1,
            ),
            RunMetricInput(
                detected=True,
                acted_within_k=False,
                detection_latency=5,
                disagreements=0,
                excerpts=1,
            ),
        ]
    )

    assert summary.median_detection_latency == 3.0
