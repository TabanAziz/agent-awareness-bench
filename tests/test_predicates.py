"""Tests for the predicate registry and its two generic factories."""

from __future__ import annotations

from typing import Any

import pytest

from awarebench.events import EventLog, EventType
from awarebench.scoring.predicates import (
    REGISTRY,
    PredicateParams,
    constant_payload_field,
    event_type_absent_since,
    register,
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
