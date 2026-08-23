"""Tests for the append-only event log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from awarebench.events import Event, EventLog, EventType, EventTypeLiteral


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _event_line(**overrides: Any) -> str:
    row: dict[str, Any] = {
        "seq": 0,
        "cycle": 0,
        "t_us": 0,
        "type": EventType.BUDGET,
        "payload": {},
    }
    row.update(overrides)
    return json.dumps(row)


def test_append_assigns_monotonic_zero_based_seq() -> None:
    log = EventLog()
    first = log.append(EventType.TOOL_CALL, cycle=0, t_us=0, payload={"call_id": "c1"})
    second = log.append(EventType.TOOL_RESULT, cycle=0, t_us=5, payload={"call_id": "c1"})
    third = log.append(EventType.MODEL_MESSAGE, cycle=1, t_us=10, payload={"role": "assistant"})

    assert first.seq == 0
    assert second.seq == 1
    assert third.seq == 2
    assert [event.seq for event in log] == [0, 1, 2]
    assert len(log) == 3


def test_event_is_frozen() -> None:
    event = Event(seq=0, cycle=0, t_us=0, type=EventType.BUDGET, payload={})

    with pytest.raises(ValidationError):
        event.seq = 7


def test_caller_dict_mutation_after_append_does_not_rewrite_history() -> None:
    log = EventLog()
    payload = {"call_id": "c1", "tool": "shell"}
    event = log.append(EventType.TOOL_CALL, cycle=0, t_us=0, payload=payload)

    payload["tool"] = "rewritten"

    assert event.payload == {"call_id": "c1", "tool": "shell"}


def test_payload_values_are_plain_python_objects() -> None:
    log = EventLog()
    event = log.append(
        EventType.REPORT,
        cycle=0,
        t_us=0,
        payload={"score": 0.75, "nested": {"a": [1, 2], "ok": True}},
    )

    assert event.payload["score"] == 0.75
    assert type(event.payload["score"]) is float
    nested = event.payload["nested"]
    assert isinstance(nested, dict)
    assert nested["ok"] is True
    items = nested["a"]
    assert isinstance(items, list)
    assert items == [1, 2]
    assert type(items[0]) is int


def test_payload_key_insertion_order_does_not_change_bytes(tmp_path: Path) -> None:
    first = EventLog()
    first.append(EventType.TOOL_CALL, 0, 0, {"call_id": "c1", "tool": "shell", "exit": 0})
    second = EventLog()
    second.append(EventType.TOOL_CALL, 0, 0, {"exit": 0, "tool": "shell", "call_id": "c1"})
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"

    first.write_jsonl(path_a)
    second.write_jsonl(path_b)

    assert path_a.read_bytes() == path_b.read_bytes()


def test_non_finite_float_payload_rejected_at_append() -> None:
    log = EventLog()

    with pytest.raises(ValueError, match="non-finite"):
        log.append(EventType.REPORT, 0, 0, {"score": float("nan")})
    with pytest.raises(ValueError, match="non-finite"):
        log.append(EventType.REPORT, 0, 0, {"score": float("inf")})

    assert len(log) == 0


def test_jsonl_roundtrip_preserves_order_fields_and_types(tmp_path: Path) -> None:
    log = EventLog()
    log.append(
        EventType.TOOL_CALL,
        cycle=0,
        t_us=100,
        payload={"call_id": "c1", "tool": "shell", "args": ["ls", "-la"]},
    )
    log.append(
        EventType.FAULT_INJECTED,
        cycle=3,
        t_us=400,
        payload={"kind": "context_truncation", "dropped_events": 2},
    )
    log.append(EventType.REPORT, cycle=4, t_us=500, payload={"score": 0.75})

    path = tmp_path / "events.jsonl"
    log.write_jsonl(path)
    loaded = EventLog.read_jsonl(path)

    assert len(loaded) == len(log)
    for original, restored in zip(log, loaded, strict=True):
        assert restored == original
        assert type(restored.seq) is int
        assert type(restored.cycle) is int
        assert type(restored.t_us) is int
        assert isinstance(restored.payload, dict)
        assert restored.payload == original.payload


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        ([_event_line(), "{not json"], "invalid event at line 2"),
        ([_event_line(), "[1, 2]"], "invalid event at line 2"),
        ([_event_line(seq=-1)], "invalid event at line 1"),
        ([_event_line(), _event_line(seq=5)], "non-monotonic seq at line 2"),
        ([_event_line(), "", _event_line(seq=1)], "blank line at line 2"),
        ([_event_line(type="teleport")], "invalid event at line 1"),
        ([_event_line(extra=1)], "invalid event at line 1"),
        (
            [_event_line(cycle=1), _event_line(seq=1, cycle=0, t_us=5)],
            "ordering violated at line 2",
        ),
    ],
)
def test_read_jsonl_rejection_paths_report_line_numbers(
    tmp_path: Path, lines: list[str], match: str
) -> None:
    path = _write_lines(tmp_path / "bad.jsonl", lines)

    with pytest.raises(ValueError, match=match):
        EventLog.read_jsonl(path)


def test_strict_int_and_key_coercions_are_rejected() -> None:
    with pytest.raises(ValueError):
        Event.model_validate({"seq": 0, "cycle": 0, "t_us": True, "type": "budget"})
    with pytest.raises(ValueError):
        Event.model_validate({"seq": 5.0, "cycle": 0, "t_us": 0, "type": "budget"})

    log = EventLog()
    bad_payload: Any = {1: "one"}
    with pytest.raises(ValueError, match="payload keys must be str"):
        log.append(EventType.REPORT, 0, 0, bad_payload)


def test_empty_log_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"

    EventLog().write_jsonl(path)

    assert path.read_text(encoding="utf-8") == ""
    assert len(EventLog.read_jsonl(path)) == 0


def test_tool_events_require_non_empty_call_id() -> None:
    log = EventLog()

    with pytest.raises(ValueError, match="call_id"):
        log.append(EventType.TOOL_CALL, 0, 0, {"tool": "shell"})
    with pytest.raises(ValueError, match="call_id"):
        log.append(EventType.TOOL_RESULT, 0, 0, {})
    with pytest.raises(ValueError, match="call_id"):
        log.append(EventType.TOOL_CALL, 0, 0, {"call_id": ""})
    with pytest.raises(ValueError, match="call_id"):
        log.append(EventType.TOOL_RESULT, 0, 0, {"call_id": ""})

    log.append(EventType.TOOL_CALL, 0, 0, {"call_id": "c1"})
    assert len(log) == 1


def test_compaction_requires_dropped_seq() -> None:
    log = EventLog()

    with pytest.raises(ValueError, match="dropped_seq"):
        log.append(EventType.COMPACTION, 0, 0, {})
    with pytest.raises(ValueError, match="dropped_seq"):
        log.append(EventType.COMPACTION, 0, 0, {"dropped_seq": [0, True]})

    log.append(EventType.COMPACTION, 0, 0, {"dropped_seq": [0, 1]})
    assert len(log) == 1


def test_append_rejects_cycle_time_regressions_but_allows_equal_and_higher() -> None:
    log = EventLog()
    log.append(EventType.BUDGET, cycle=0, t_us=10, payload={})

    with pytest.raises(ValueError, match="precedes previous"):
        log.append(EventType.BUDGET, cycle=0, t_us=5, payload={})
    assert len(log) == 1

    log.append(EventType.BUDGET, cycle=0, t_us=10, payload={})
    log.append(EventType.BUDGET, cycle=1, t_us=0, payload={})
    assert len(log) == 3


def test_event_type_constants_match_the_literal_union() -> None:
    constants = {
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.MODEL_MESSAGE,
        EventType.COMPACTION,
        EventType.FAULT_INJECTED,
        EventType.BUDGET,
        EventType.REPORT,
    }

    assert constants == set(get_args(EventTypeLiteral))
    assert EventType.TOOL_CALL == "tool_call"
    assert EventType.TOOL_RESULT == "tool_result"
    assert EventType.MODEL_MESSAGE == "model_message"
    assert EventType.COMPACTION == "compaction"
    assert EventType.FAULT_INJECTED == "fault_injected"
    assert EventType.BUDGET == "budget"
    assert EventType.REPORT == "report"


def test_package_reexports_typed_event_helpers() -> None:
    import awarebench

    assert awarebench.EventTypeLiteral is EventTypeLiteral
    assert awarebench.Event is Event
