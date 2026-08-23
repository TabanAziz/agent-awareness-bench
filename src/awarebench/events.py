"""Append-only event log shared by all harness modules."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)


class EventType:
    """String constants for event types.

    Each constant must equal its counterpart in EventTypeLiteral; the
    correspondence is pinned by tests.
    """

    TOOL_CALL: Final = "tool_call"
    TOOL_RESULT: Final = "tool_result"
    MODEL_MESSAGE: Final = "model_message"
    COMPACTION: Final = "compaction"
    FAULT_INJECTED: Final = "fault_injected"
    BUDGET: Final = "budget"
    REPORT: Final = "report"


EventTypeLiteral = Literal[
    "tool_call",
    "tool_result",
    "model_message",
    "compaction",
    "fault_injected",
    "budget",
    "report",
]

# PEP 695 recursive alias: native syntax gives pydantic a resolvable recursive
# schema while storing plain Python values, and mypy resolves the self-reference
# natively. Both wrapper-based and quoted-forward-ref forms fail one toolchain
# or the other on older interpreters; this is why the floor is Python 3.12.
type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


def _reject_non_finite_floats(node: object) -> None:
    """Raise ValueError when node contains a NaN or infinite float at any depth."""
    if isinstance(node, float):
        if not math.isfinite(node):
            raise ValueError("payload contains non-finite float (nan/inf)")
    elif isinstance(node, dict):
        for value in node.values():
            _reject_non_finite_floats(value)
    elif isinstance(node, list):
        for item in node:
            _reject_non_finite_floats(item)


class Event(BaseModel):
    """Single immutable log entry.

    seq, cycle, and t_us are strict ints; bools and floats are rejected instead
    of coerced. t_us is virtual time in integer microseconds, never float.
    payload holds JSON values only and is validated at construction time, so
    the append and read paths enforce identical rules.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: StrictInt = Field(ge=0)
    cycle: StrictInt = Field(ge=0)
    t_us: StrictInt = Field(ge=0)
    type: EventTypeLiteral
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("payload", mode="before")
    @classmethod
    def _reject_bad_payload(cls, value: object) -> object:
        """Reject non-str keys (pydantic would coerce them silently) and non-finite floats."""
        if not isinstance(value, dict):
            return value
        for key in value:
            if not isinstance(key, str):
                raise ValueError(f"payload keys must be str, got {type(key).__name__}")
        _reject_non_finite_floats(value)
        return value

    @model_validator(mode="after")
    def _enforce_typed_payload_minimums(self) -> Event:
        """Enforce typed minimums for structured event types.

        TOOL_CALL and TOOL_RESULT carry call_id: str; COMPACTION carries
        dropped_seq: list[int]. All other event types stay free-form JSON dicts.
        """
        if self.type in (EventType.TOOL_CALL, EventType.TOOL_RESULT):
            call_id = self.payload.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(
                    f"event type '{self.type}' requires non-empty payload field call_id: str"
                )
        elif self.type == EventType.COMPACTION:
            dropped_seq = self.payload.get("dropped_seq")
            if not isinstance(dropped_seq, list) or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in dropped_seq
            ):
                raise ValueError(
                    f"event type '{self.type}' requires payload field dropped_seq: list[int]"
                )
        return self


class EventLog:
    """Append-only sequence of events with monotonic zero-based seq numbers.

    Guarantees: payloads are deep-copied at append time so later mutation of
    the caller's dict cannot rewrite history; entries are frozen once created;
    (cycle, t_us) pairs are monotonically non-decreasing across appends.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []

    def append(
        self, event_type: EventTypeLiteral, cycle: int, t_us: int, payload: dict[str, Any]
    ) -> Event:
        """Append a new event with the next seq number and return it.

        The payload is deep-copied at append time; mutating the caller's dict
        afterwards cannot alter the stored entry, and entries themselves are
        frozen. Raises ValueError if (cycle, t_us) is lexicographically less
        than the previous event's pair, or if the event fails validation.
        """
        if self._events:
            last = self._events[-1]
            if (cycle, t_us) < (last.cycle, last.t_us):
                raise ValueError(
                    f"(cycle, t_us)=({cycle}, {t_us}) precedes previous "
                    f"(cycle, t_us)=({last.cycle}, {last.t_us}) at seq {last.seq}"
                )
        event = Event(
            seq=len(self._events),
            cycle=cycle,
            t_us=t_us,
            type=event_type,
            payload=copy.deepcopy(payload),
        )
        self._events.append(event)
        return event

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def write_jsonl(self, path: str | Path) -> None:
        """Write all events to path as JSON lines, one event per line.

        Every line is serialized before the file is opened, so a serialization
        failure never leaves a partial file behind.
        """
        lines = [
            json.dumps(
                event.model_dump(mode="json"),
                sort_keys=True,
                allow_nan=False,
                ensure_ascii=True,
            )
            for event in self._events
        ]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))

    @classmethod
    def read_jsonl(cls, path: str | Path) -> EventLog:
        """Load events from a JSONL file written by write_jsonl.

        Raises ValueError, reporting the offending line number, on malformed
        JSON, non-object lines, interior blank lines, out-of-order or duplicate
        seq, and (cycle, t_us) regressions. Trailing whitespace at EOF is
        tolerated.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()
        end = len(raw_lines)
        while end > 0 and not raw_lines[end - 1].strip():
            end -= 1
        parsed: list[tuple[int, Event]] = []
        for line_number, line in enumerate(raw_lines[:end], start=1):
            if not line.strip():
                raise ValueError(f"blank line at line {line_number}")
            try:
                event = Event.model_validate(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"invalid event at line {line_number}: {exc}") from exc
            parsed.append((line_number, event))
        previous: Event | None = None
        for expected_seq, (line_number, event) in enumerate(parsed):
            if event.seq != expected_seq:
                raise ValueError(
                    f"non-monotonic seq at line {line_number}: "
                    f"got {event.seq}, expected {expected_seq}"
                )
            if previous is not None and (event.cycle, event.t_us) < (
                previous.cycle,
                previous.t_us,
            ):
                raise ValueError(
                    f"(cycle, t_us) ordering violated at line {line_number}: "
                    f"(cycle={event.cycle}, t_us={event.t_us}) precedes "
                    f"(cycle={previous.cycle}, t_us={previous.t_us})"
                )
            previous = event
        log = cls()
        log._events = [event for _, event in parsed]
        return log
