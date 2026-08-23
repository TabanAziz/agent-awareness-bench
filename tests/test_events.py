"""Tests for the append-only event log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from awarebench.events import Event, EventLog, EventType


def test_append_assigns_monotonic_zero_based_seq() -> None:
    log = EventLog()
    first = log.append(EventType.TOOL_CALL, cycle=0, t_us=0, payload={"tool": "shell"})
    second = log.append(EventType.TOOL_RESULT, cycle=0, t_us=5, payload={"exit_code": 0})
    third = log.append(
        EventType.MODEL_MESSAGE, cycle=1, t_us=10, payload={"role": "assistant"}
    )

    assert first.seq == 0
    assert second.seq == 1
    assert third.seq == 2
    assert [event.seq for event in log] == [0, 1, 2]
    assert len(log) == 3


def test_event_is_frozen() -> None:
    event = Event(seq=0, cycle=0, t_us=0, type=EventType.BUDGET, payload={})

    with pytest.raises(ValidationError):
        event.seq = 7


def test_jsonl_roundtrip_preserves_order_fields_and_types(tmp_path: Path) -> None:
    log = EventLog()
    log.append(
        EventType.TOOL_CALL,
        cycle=0,
        t_us=100,
        payload={"tool": "shell", "args": ["ls", "-la"]},
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


def test_read_jsonl_rejects_out_of_order_seq(tmp_path: Path) -> None:
    path = tmp_path / "out_of_order.jsonl"
    rows = [
        {"seq": 0, "cycle": 0, "t_us": 0, "type": EventType.BUDGET, "payload": {}},
        {"seq": 2, "cycle": 1, "t_us": 9, "type": EventType.REPORT, "payload": {}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-monotonic seq"):
        EventLog.read_jsonl(path)


def test_read_jsonl_rejects_duplicate_seq(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    rows = [
        {"seq": 0, "cycle": 0, "t_us": 0, "type": EventType.BUDGET, "payload": {}},
        {"seq": 0, "cycle": 0, "t_us": 1, "type": EventType.BUDGET, "payload": {}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-monotonic seq"):
        EventLog.read_jsonl(path)


def test_event_type_constants_are_stable_strings() -> None:
    assert EventType.TOOL_CALL == "tool_call"
    assert EventType.TOOL_RESULT == "tool_result"
    assert EventType.MODEL_MESSAGE == "model_message"
    assert EventType.COMPACTION == "compaction"
    assert EventType.FAULT_INJECTED == "fault_injected"
    assert EventType.BUDGET == "budget"
    assert EventType.REPORT == "report"
