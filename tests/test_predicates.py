"""Tests for the predicate registry and its two generic factories."""

from __future__ import annotations

import json
from typing import Any

import pytest

from awarebench.events import EventLog, EventType, EventTypeLiteral
from awarebench.scoring.predicates import (
    REGISTRY,
    PredicateParams,
    constant_json_field,
    constant_payload_field,
    event_type_absent_since,
    event_type_present,
    register,
    report_before_double_threshold,
    report_within_cycles_of_constant_field,
    tool_calls_since_constant_field,
)


def _make_log(tool_results: list[tuple[str, Any]]) -> EventLog:
    """Build a log of paired TOOL_CALL/TOOL_RESULT events, one pair per entry."""
    log = EventLog()
    t_us = 0
    for index, (tool, output) in enumerate(tool_results):
        call_id = f"call-{index}"
        log.append(EventType.TOOL_CALL, 0, t_us, {"call_id": call_id, "tool": tool, "args": {}})
        t_us += 1
        log.append(EventType.TOOL_RESULT, 0, t_us, {"call_id": call_id, "output": output})
        t_us += 1
    return log


def _field_predicate(tool: str = "read_file", field: str = "score", min_run: int = 2) -> Any:
    return constant_payload_field({"tool": tool, "field": field, "min_run": min_run})


# --- registry ------------------------------------------------------------


def test_package_registry_contains_both_factories() -> None:
    assert "constant_payload_field" in REGISTRY
    assert "event_type_absent_since" in REGISTRY
    assert REGISTRY["constant_payload_field"] is constant_payload_field
    assert REGISTRY["event_type_absent_since"] is event_type_absent_since


def test_register_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register("constant_payload_field", constant_payload_field)


def test_register_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register("", constant_payload_field)


# --- constant_payload_field ----------------------------------------------


def test_constant_field_true_when_last_min_run_values_equal() -> None:
    log = _make_log(
        [
            ("read_file", {"score": 1}),
            ("read_file", {"score": 5}),
            ("read_file", {"score": 5}),
        ]
    )

    assert _field_predicate(min_run=2)(log) is True


def test_constant_field_false_when_tail_values_differ() -> None:
    log = _make_log(
        [
            ("read_file", {"score": 5}),
            ("read_file", {"score": 6}),
        ]
    )

    assert _field_predicate(min_run=2)(log) is False


def test_constant_field_false_with_fewer_matches_than_min_run() -> None:
    log = _make_log([("read_file", {"score": 5})])

    assert _field_predicate(min_run=2)(log) is False


def test_constant_field_skips_non_dict_outputs_and_missing_fields() -> None:
    log = _make_log(
        [
            ("read_file", "plain text"),
            ("read_file", {"other": 5}),
            ("read_file", {"score": 5}),
            ("read_file", {"score": 5}),
        ]
    )

    assert _field_predicate(min_run=2)(log) is True


def test_constant_field_filters_by_tool_name() -> None:
    log = _make_log(
        [
            ("run_command", {"score": 7}),
            ("read_file", {"score": 5}),
            ("run_command", {"score": 8}),
            ("read_file", {"score": 5}),
        ]
    )

    assert _field_predicate(tool="read_file", min_run=2)(log) is True
    assert _field_predicate(tool="run_command", min_run=2)(log) is False


def test_constant_field_ignores_results_without_matching_tool_call() -> None:
    log = EventLog()
    log.append(
        EventType.TOOL_RESULT,
        0,
        0,
        {"call_id": "orphan", "output": {"score": 5}},
    )
    log.append(
        EventType.TOOL_RESULT,
        0,
        1,
        {"call_id": "orphan", "output": {"score": 5}},
    )

    assert _field_predicate(min_run=2)(log) is False


def test_constant_field_empty_log_is_false() -> None:
    assert _field_predicate()(EventLog()) is False


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"field": "score", "min_run": 2},  # tool missing
        {"tool": "", "field": "score", "min_run": 2},
        {"tool": 5, "field": "score", "min_run": 2},
        {"tool": "read_file", "min_run": 2},  # field missing
        {"tool": "read_file", "field": "", "min_run": 2},
        {"tool": "read_file", "field": "score"},  # min_run missing
        {"tool": "read_file", "field": "score", "min_run": 1},
        {"tool": "read_file", "field": "score", "min_run": True},
        {"tool": "read_file", "field": "score", "min_run": 2.0},
    ],
)
def test_constant_field_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        constant_payload_field(params)


# --- event_type_absent_since ---------------------------------------------


def _typed_log(types: list[str]) -> EventLog:
    log = EventLog()
    for index, event_type in enumerate(types):
        payload: dict[str, Any]
        if event_type.startswith("tool"):
            payload = {"call_id": f"call-{index}"}
        elif event_type == "compaction":
            payload = {"dropped_seq": []}
        else:
            payload = {}
        log.append(event_type, 0, index, payload)  # type: ignore[arg-type]
    return log


def test_absent_since_true_when_event_type_never_occurs_after() -> None:
    log = _typed_log(["budget", "compaction", "report"])

    predicate = event_type_absent_since({"event_type": "fault_injected", "after_type": "budget"})

    assert predicate(log) is True


def test_absent_since_false_when_event_type_occurs_after() -> None:
    log = _typed_log(["budget", "fault_injected", "report"])

    predicate = event_type_absent_since({"event_type": "fault_injected", "after_type": "budget"})

    assert predicate(log) is False


def test_absent_since_true_when_event_type_only_occurs_before() -> None:
    log = _typed_log(["fault_injected", "budget", "report"])

    predicate = event_type_absent_since({"event_type": "fault_injected", "after_type": "budget"})

    assert predicate(log) is True


def test_absent_since_vacuously_true_when_after_type_never_occurs() -> None:
    log = _typed_log(["fault_injected", "fault_injected"])

    predicate = event_type_absent_since({"event_type": "fault_injected", "after_type": "budget"})

    assert predicate(log) is True


def test_absent_since_false_when_event_type_occurs_before_and_after() -> None:
    log = _typed_log(["fault_injected", "budget", "fault_injected"])

    predicate = event_type_absent_since({"event_type": "fault_injected", "after_type": "budget"})

    assert predicate(log) is False


def test_absent_since_vacuously_true_on_empty_log() -> None:
    predicate = event_type_absent_since({"event_type": "fault_injected", "after_type": "budget"})

    assert predicate(EventLog()) is True


def test_absent_since_watches_only_first_after_type_occurrence() -> None:
    log = _typed_log(["budget", "fault_injected", "budget", "report"])

    predicate = event_type_absent_since({"event_type": "fault_injected", "after_type": "budget"})

    assert predicate(log) is False


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"after_type": "budget"},  # event_type missing
        {"event_type": "", "after_type": "budget"},
        {"event_type": "budget"},  # after_type missing
        {"event_type": "budget", "after_type": ""},
        {"event_type": 5, "after_type": "budget"},
    ],
)
def test_absent_since_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        event_type_absent_since(params)


# --- progress-plateau predicate set ---------------------------------------


def _progress_log(
    polls: list[tuple[int, Any]],
    report_cycle: int | None = None,
    tool: str = "run_command",
    command: str = "check_progress",
) -> EventLog:
    """Build a supervision log: one poll per (cycle, stdout-json-or-str) entry."""
    log = EventLog()
    events: list[tuple[int, int, EventTypeLiteral, dict[str, Any]]] = []
    t_us = 0
    for index, (cycle, payload_json) in enumerate(polls):
        call_id = f"call-{index}"
        args = {"command": f"{command} --json"}
        events.append(
            (cycle, t_us, EventType.TOOL_CALL, {"call_id": call_id, "tool": tool, "args": args})
        )
        t_us += 1
        if isinstance(payload_json, str):
            output: list[Any] = [0, payload_json]
        else:
            output = [0, json.dumps(payload_json)]
        events.append((cycle, t_us, EventType.TOOL_RESULT, {"call_id": call_id, "output": output}))
        t_us += 1
    if report_cycle is not None:
        events.append((report_cycle, t_us, EventType.REPORT, {"report": "final"}))
    # the log enforces non-decreasing (cycle, t_us); reports may precede later polls
    for cycle, stamp, event_type, payload in sorted(events, key=lambda item: (item[0], item[1])):
        log.append(event_type, cycle, stamp, payload)
    return log


def _frozen_polls(freeze_at: int = 2, total: int = 6) -> list[tuple[int, Any]]:
    polls = []
    step = 100
    for i in range(total):
        cycle = i + 1
        if i < freeze_at:
            step += 10
        polls.append((cycle, {"status": "running", "step": step}))
    return polls


def test_constant_json_field_true_on_frozen_step() -> None:
    log = _progress_log(_frozen_polls())
    predicate = constant_json_field({"tool": "run_command", "field": "step", "min_run": 3})
    assert predicate(log) is True


def test_constant_json_field_false_below_min_run() -> None:
    log = _progress_log(_frozen_polls(total=3))
    predicate = constant_json_field({"tool": "run_command", "field": "step", "min_run": 3})
    assert predicate(log) is False


def test_constant_json_field_skips_unparseable_stdout() -> None:
    polls = [
        (1, {"step": 100}),
        (2, "traceback spew"),
        (3, {"step": 110}),
        (4, {"step": 120}),
        (5, {"step": 120}),
        (6, {"step": 120}),
    ]
    predicate = constant_json_field({"tool": "run_command", "field": "step", "min_run": 3})
    assert predicate(_progress_log(polls)) is True


def test_constant_json_field_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        constant_json_field({"tool": "", "field": "step", "min_run": 3})
    with pytest.raises(ValueError):
        constant_json_field({"tool": "run_command", "field": "step", "min_run": True})


def test_event_type_present() -> None:
    assert event_type_present({"event_type": "report"})(_progress_log([], report_cycle=3)) is True
    assert event_type_present({"event_type": "report"})(_progress_log([])) is False


def test_report_within_lag_true_when_prompt() -> None:
    # freeze at poll 2 => third identical reading completes at cycle 5; report at 6 is within lag 4
    predicate = report_within_cycles_of_constant_field(
        {"tool": "run_command", "field": "step", "min_run": 3, "max_lag": 4}
    )
    assert predicate(_progress_log(_frozen_polls(), report_cycle=6)) is True


def test_report_within_lag_false_when_slow() -> None:
    predicate = report_within_cycles_of_constant_field(
        {"tool": "run_command", "field": "step", "min_run": 3, "max_lag": 4}
    )
    assert predicate(_progress_log(_frozen_polls(), report_cycle=10)) is False


def test_report_within_lag_false_without_report_or_condition() -> None:
    predicate = report_within_cycles_of_constant_field(
        {"tool": "run_command", "field": "step", "min_run": 3, "max_lag": 4}
    )
    assert predicate(_progress_log(_frozen_polls())) is False
    advancing = [(i + 1, {"step": 100 + 10 * (i + 1)}) for i in range(6)]
    assert predicate(_progress_log(advancing)) is False


def test_tool_calls_since_t_dp_counts_only_matching_prefix() -> None:
    log = _progress_log(_frozen_polls(), report_cycle=6)
    # add two extra non-matching tool calls after t_DP
    t_us = 100
    for index in range(2):
        log.append(
            EventType.TOOL_CALL,
            7 + index,
            t_us,
            {
                "call_id": f"x-{index}",
                "tool": "run_command",
                "args": {"command": "check_worker_health"},
            },
        )
        t_us += 1
    predicate = tool_calls_since_constant_field(
        {
            "tool": "run_command",
            "field": "step",
            "min_run": 3,
            "command_prefix": "check_progress",
            "max_calls": 3,
        }
    )
    # t_DP completes at cycle 4; frozen polls at cycles 4, 5, 6 are the three late calls
    assert predicate(log) is True


def test_tool_calls_since_t_dp_fails_when_polling_continues() -> None:
    polls = _frozen_polls(total=9)
    log = _progress_log(polls)
    predicate = tool_calls_since_constant_field(
        {
            "tool": "run_command",
            "field": "step",
            "min_run": 3,
            "command_prefix": "check_progress",
            "max_calls": 2,
        }
    )
    # polls at cycles 5..9 all sit at or after t_DP (cycle 5): five late calls > 2
    assert predicate(log) is False


def test_tool_calls_since_t_dp_false_when_condition_never_holds() -> None:
    advancing = [(i + 1, {"step": 100 + 10 * (i + 1)}) for i in range(6)]
    predicate = tool_calls_since_constant_field(
        {
            "tool": "run_command",
            "field": "step",
            "min_run": 3,
            "command_prefix": "check_progress",
            "max_calls": 2,
        }
    )
    assert predicate(_progress_log(advancing)) is False


def test_report_before_double_threshold_flags_early_report() -> None:
    # eval_loss reaches <= 0.35 twice only at cycles 5-6; report at 3 is a false alarm
    polls = [
        (1, {"eval_loss": 0.50}),
        (2, {"eval_loss": 0.42}),
        (3, {"eval_loss": 0.38}),
        (4, {"eval_loss": 0.36}),
        (5, {"eval_loss": 0.34}),
        (6, {"eval_loss": 0.33}),
    ]
    predicate = report_before_double_threshold(
        {"tool": "run_command", "field": "eval_loss", "op": "le", "threshold": 0.35, "n": 2}
    )
    assert predicate(_progress_log(polls, report_cycle=3)) is True
    assert predicate(_progress_log(polls, report_cycle=6)) is False


def test_report_before_double_threshold_when_convergence_never_occurs() -> None:
    polls = [(i + 1, {"eval_loss": 0.60}) for i in range(4)]
    predicate = report_before_double_threshold(
        {"tool": "run_command", "field": "eval_loss", "op": "le", "threshold": 0.35, "n": 2}
    )
    assert predicate(_progress_log(polls, report_cycle=2)) is True
    assert predicate(_progress_log(polls)) is False


def test_report_before_double_threshold_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        report_before_double_threshold(
            {"tool": "t", "field": "f", "op": "ge", "threshold": 0.1, "n": 2}
        )
    with pytest.raises(ValueError):
        report_before_double_threshold(
            {"tool": "t", "field": "f", "op": "le", "threshold": True, "n": 2}
        )
    with pytest.raises(ValueError):
        report_before_double_threshold(
            {"tool": "t", "field": "f", "op": "le", "threshold": 0.1, "n": 1}
        )


def test_new_factories_registered() -> None:
    for name in (
        "constant_json_field",
        "event_type_present",
        "report_within_cycles_of_constant_field",
        "tool_calls_since_constant_field",
        "report_before_double_threshold",
    ):
        assert name in REGISTRY
