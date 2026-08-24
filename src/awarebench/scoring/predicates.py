"""Registry of machine-checkable success predicates over event logs.

A predicate factory validates its params at factory time (ValueError on bad
params) and returns a pure, deterministic function of an EventLog. Registries
are plain dicts so probe loaders can inject subsets; the package-level
REGISTRY is the default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from awarebench.events import EventLog, EventType, JsonValue

PredicateParams = dict[str, JsonValue]
LogPredicate = Callable[[EventLog], bool]
PredicateFactory = Callable[[PredicateParams], LogPredicate]

REGISTRY: Final[dict[str, PredicateFactory]] = {}


def register(name: str, factory: PredicateFactory) -> PredicateFactory:
    """Register factory under name and return it unchanged.

    Raises ValueError on empty names and duplicate registrations; a silent
    overwrite would let two different checks masquerade as one predicate.
    """
    if not name:
        raise ValueError("predicate name must be non-empty")
    if name in REGISTRY:
        raise ValueError(f"predicate '{name}' is already registered")
    REGISTRY[name] = factory
    return factory


def _require_non_empty_str(value: JsonValue | None, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_min_run(value: JsonValue | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"min_run must be an int, got {type(value).__name__}")
    if value < 2:
        raise ValueError(f"min_run must be >= 2, got {value}")
    return value


def constant_payload_field(params: PredicateParams) -> LogPredicate:
    """True when the last min_run field values of one tool's outputs all agree.

    Correlates TOOL_RESULT events to their TOOL_CALL via call_id, keeps
    results of the named tool whose payload["output"] is a dict containing
    "field", and returns True iff at least min_run such values exist and the
    last min_run are all equal. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    min_run = _require_min_run(params.get("min_run"))

    def predicate(log: EventLog) -> bool:
        tool_by_call: dict[str, str] = {}
        for event in log:
            if event.type != EventType.TOOL_CALL:
                continue
            call_id = event.payload.get("call_id")
            tool_name = event.payload.get("tool")
            if isinstance(call_id, str) and isinstance(tool_name, str):
                tool_by_call[call_id] = tool_name
        values: list[JsonValue] = []
        for event in log:
            if event.type != EventType.TOOL_RESULT:
                continue
            call_id = event.payload.get("call_id")
            if not isinstance(call_id, str) or tool_by_call.get(call_id) != tool:
                continue
            output = event.payload.get("output")
            if isinstance(output, dict) and field in output:
                values.append(output[field])
        if len(values) < min_run:
            return False
        tail = values[-min_run:]
        last = tail[-1]
        return all(value == last for value in tail)

    return predicate


def event_type_absent_since(params: PredicateParams) -> LogPredicate:
    """True when event_type never occurs after the first after_type event.

    When after_type never occurs the watched window is empty and the
    predicate holds vacuously. Deterministic and pure.
    """
    event_type = _require_non_empty_str(params.get("event_type"), "event_type")
    after_type = _require_non_empty_str(params.get("after_type"), "after_type")

    def predicate(log: EventLog) -> bool:
        seen_after = False
        for event in log:
            if not seen_after:
                seen_after = event.type == after_type
            elif event.type == event_type:
                return False
        return True

    return predicate


register("constant_payload_field", constant_payload_field)
register("event_type_absent_since", event_type_absent_since)
