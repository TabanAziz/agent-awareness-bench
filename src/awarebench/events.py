"""Append-only event log shared by all harness modules."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType:
    """String constants for event types."""

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MODEL_MESSAGE = "model_message"
    COMPACTION = "compaction"
    FAULT_INJECTED = "fault_injected"
    BUDGET = "budget"
    REPORT = "report"


class Event(BaseModel):
    """Single immutable log entry.

    t_us is virtual time in integer microseconds; floats are never used for time.
    """

    model_config = ConfigDict(frozen=True)

    seq: int = Field(ge=0)
    cycle: int = Field(ge=0)
    t_us: int = Field(ge=0)
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class EventLog:
    """Append-only sequence of events with monotonic zero-based seq numbers."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def append(
        self, event_type: str, cycle: int, t_us: int, payload: dict[str, Any]
    ) -> Event:
        """Append a new event with the next seq number and return it."""
        event = Event(
            seq=len(self._events), cycle=cycle, t_us=t_us, type=event_type, payload=payload
        )
        self._events.append(event)
        return event

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def write_jsonl(self, path: str | Path) -> None:
        """Write all events to path as JSON lines, one event per line."""
        with Path(path).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(event.model_dump()) + "\n" for event in self._events)

    @classmethod
    def read_jsonl(cls, path: str | Path) -> EventLog:
        """Load events from a JSONL file written by write_jsonl.

        Raises ValueError if any seq is out of order or duplicated.
        """
        events: list[Event] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(Event.model_validate(json.loads(stripped)))
                except ValueError as exc:
                    raise ValueError(f"invalid event at line {line_number}: {exc}") from exc
        for expected_seq, event in enumerate(events):
            if event.seq != expected_seq:
                raise ValueError(
                    f"non-monotonic seq at position {expected_seq}: got {event.seq}"
                )
        log = cls()
        log._events = events
        return log
