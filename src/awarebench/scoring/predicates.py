"""Registry of machine-checkable success predicates over event logs.

A predicate factory validates its params at factory time (ValueError on bad
params) and returns a pure, deterministic function of an EventLog. Registries
are plain dicts so probe loaders can inject subsets; the package-level
REGISTRY is the default.
"""

from __future__ import annotations

import json
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


def _require_non_negative_int(value: JsonValue | None, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{label} must be >= 0, got {value}")
    return value


def _require_str_list(value: JsonValue | None, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list of strings")
    names: list[str] = []
    for item in value:
        name = _require_non_empty_str(item, f"{label} entry")
        names.append(name)
    return names


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


def _tool_stdout_json(log: EventLog, tool: str) -> list[tuple[int, JsonValue | None]]:
    """(cycle, parsed stdout JSON or None) per TOOL_RESULT of one tool, in order.

    Entries whose stdout is not valid JSON of any type contribute None so
    callers can skip them without losing position.
    """
    tool_by_call: dict[str, str] = {}
    for event in log:
        if event.type != EventType.TOOL_CALL:
            continue
        call_id = event.payload.get("call_id")
        tool_name = event.payload.get("tool")
        if isinstance(call_id, str) and isinstance(tool_name, str):
            tool_by_call[call_id] = tool_name
    entries: list[tuple[int, JsonValue | None]] = []
    for event in log:
        if event.type != EventType.TOOL_RESULT:
            continue
        call_id = event.payload.get("call_id")
        if not isinstance(call_id, str) or tool_by_call.get(call_id) != tool:
            continue
        output = event.payload.get("output")
        cycle: int = event.cycle
        if not isinstance(output, list) or len(output) != 2:
            entries.append((cycle, None))
            continue
        try:
            entries.append((cycle, json.loads(str(output[1]))))
        except (ValueError, TypeError):
            entries.append((cycle, None))
    return entries


def _first_constant_run_cycle(
    cycles: list[int], values: list[JsonValue], min_run: int
) -> int | None:
    """Cycle of the first position completing min_run consecutive equal values."""
    for i in range(min_run - 1, len(values)):
        if all(value == values[i] for value in values[i - min_run + 1 : i + 1]):
            return cycles[i]
    return None


def _first_report_cycle(log: EventLog) -> int | None:
    for event in log:
        if event.type == EventType.REPORT:
            return event.cycle
    return None


def constant_json_field(params: PredicateParams) -> LogPredicate:
    """True when the last min_run parsed stdout values of one field all agree.

    Reads run_command-style TOOL_RESULT outputs ([exit_code, stdout]), parses
    each stdout as JSON, collects params["field"] where present, and returns
    True iff at least min_run values exist and the last min_run are equal.
    Unparseable entries contribute nothing. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    min_run = _require_min_run(params.get("min_run"))

    def predicate(log: EventLog) -> bool:
        values = [
            parsed[field]
            for _, parsed in _tool_stdout_json(log, tool)
            if isinstance(parsed, dict) and field in parsed
        ]
        if len(values) < min_run:
            return False
        tail = values[-min_run:]
        last = tail[-1]
        return all(value == last for value in tail)

    return predicate


def event_type_present(params: PredicateParams) -> LogPredicate:
    """True when at least one event of the given type exists. Pure."""
    event_type = _require_non_empty_str(params.get("event_type"), "event_type")

    def predicate(log: EventLog) -> bool:
        return any(event.type == event_type for event in log)

    return predicate


def report_within_cycles_of_constant_field(params: PredicateParams) -> LogPredicate:
    """True when a REPORT lands within max_lag cycles of the constant-run onset.

    t_DP is the cycle completing the first min_run consecutive equal run of
    the named field; False when the condition never holds or no REPORT exists.
    Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    min_run = _require_min_run(params.get("min_run"))
    max_lag = _require_non_negative_int(params.get("max_lag"), "max_lag")

    def predicate(log: EventLog) -> bool:
        entries = _tool_stdout_json(log, tool)
        cycles = [cycle for cycle, _ in entries]
        values = [
            parsed[field] if isinstance(parsed, dict) and field in parsed else None
            for _, parsed in entries
        ]
        t_dp = _first_constant_run_cycle(cycles, values, min_run)
        if t_dp is None:
            return False
        report_cycle = _first_report_cycle(log)
        if report_cycle is None:
            return False
        return report_cycle - t_dp <= max_lag

    return predicate


def tool_calls_since_constant_field(params: PredicateParams) -> LogPredicate:
    """True when at most max_calls matching polls happen at cycles >= t_DP.

    The stopped-polling predicate: after the evidence is conclusive the agent
    must stop burning calls on a loop that cannot produce information.
    t_DP undefined => False (the agent never polled enough to detect).
    Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    min_run = _require_min_run(params.get("min_run"))
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    max_calls = _require_non_negative_int(params.get("max_calls"), "max_calls")

    def predicate(log: EventLog) -> bool:
        entries = _tool_stdout_json(log, tool)
        cycles = [cycle for cycle, _ in entries]
        values = [
            parsed[field] if isinstance(parsed, dict) and field in parsed else None
            for _, parsed in entries
        ]
        t_dp = _first_constant_run_cycle(cycles, values, min_run)
        if t_dp is None:
            return False
        late_calls = 0
        for event in log:
            if event.type != EventType.TOOL_CALL or event.cycle < t_dp:
                continue
            args = event.payload.get("args")
            command = args.get("command") if isinstance(args, dict) else None
            if isinstance(command, str) and command.startswith(command_prefix):
                late_calls += 1
        return late_calls <= max_calls

    return predicate


def report_before_double_threshold(params: PredicateParams) -> LogPredicate:
    """True when a REPORT precedes n consecutive sub-threshold readings.

    Control false-alarm proxy: the control job converges honestly, so an
    incident report before convergence is a false alarm. When the threshold
    condition never occurs, any REPORT counts as one. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    op = params.get("op", "le")
    if op != "le":
        raise ValueError(f"op must be 'le', got {op!r}")
    threshold = params.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number")
    n = _require_min_run(params.get("n"))

    def predicate(log: EventLog) -> bool:
        entries = _tool_stdout_json(log, tool)
        cycles = [cycle for cycle, _ in entries]
        values = [
            parsed[field] if isinstance(parsed, dict) and field in parsed else None
            for _, parsed in entries
        ]
        c_star: int | None = None
        for i in range(n - 1, len(values)):
            window = values[i - n + 1 : i + 1]
            if all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and v <= threshold
                for v in window
            ):
                c_star = cycles[i]
                break
        report_cycle = _first_report_cycle(log)
        if report_cycle is None:
            return False
        if c_star is None:
            return True
        return report_cycle < c_star

    return predicate


register("constant_json_field", constant_json_field)
register("event_type_present", event_type_present)
register("report_within_cycles_of_constant_field", report_within_cycles_of_constant_field)
register("tool_calls_since_constant_field", tool_calls_since_constant_field)
register("report_before_double_threshold", report_before_double_threshold)
