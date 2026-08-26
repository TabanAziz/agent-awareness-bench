"""Registry of machine-checkable success predicates over event logs.

A predicate factory validates its params at factory time (ValueError on bad
params) and returns a pure, deterministic function of an EventLog. Registries
are plain dicts so probe loaders can inject subsets; the package-level
REGISTRY is the default.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Final

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


def _field_series(log: EventLog, tool: str, field: str) -> list[tuple[int, JsonValue]]:
    """(cycle, value) per TOOL_RESULT of one tool whose stdout JSON has the field.

    Entries without the field (other commands, unparseable stdout) contribute
    nothing at all - they must never pad the sequence, or three consecutive
    field-absent polls would masquerade as a constant run.
    """
    return [
        (cycle, parsed[field])
        for cycle, parsed in _tool_stdout_json(log, tool)
        if isinstance(parsed, dict) and field in parsed
    ]


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
    the named field; the report must come AT or AFTER t_DP (a report before
    any evidence is a guess, not detection) and within max_lag cycles of it.
    False when the condition never holds or no REPORT exists. Deterministic
    and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    min_run = _require_min_run(params.get("min_run"))
    max_lag = _require_non_negative_int(params.get("max_lag"), "max_lag")

    def predicate(log: EventLog) -> bool:
        series = _field_series(log, tool, field)
        cycles = [cycle for cycle, _ in series]
        values = [value for _, value in series]
        t_dp = _first_constant_run_cycle(cycles, values, min_run)
        if t_dp is None:
            return False
        report_cycle = _first_report_cycle(log)
        if report_cycle is None:
            return False
        return 0 <= report_cycle - t_dp <= max_lag

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
        series = _field_series(log, tool, field)
        cycles = [cycle for cycle, _ in series]
        values = [value for _, value in series]
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
    """Report-before-convergence check; invertible for control scoring.

    Raw semantics (negate=false): True when a REPORT precedes n consecutive
    sub-threshold readings - i.e. a false alarm on a converging job. When the
    threshold condition never occurs, any REPORT counts as one, and no REPORT
    is never a false alarm. With negate=true the result flips, so control runs
    pass when the agent reports only after honest convergence. Deterministic
    and pure.
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
    negate = params.get("negate", False)
    if not isinstance(negate, bool):
        raise ValueError("negate must be a bool")

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
            raw = False
        elif c_star is None:
            raw = True
        else:
            raw = report_cycle < c_star
        return raw != negate

    return predicate


register("constant_json_field", constant_json_field)
register("event_type_present", event_type_present)
register("report_within_cycles_of_constant_field", report_within_cycles_of_constant_field)
register("tool_calls_since_constant_field", tool_calls_since_constant_field)
register("report_before_double_threshold", report_before_double_threshold)


# --- wave-1 shared helpers --------------------------------------------------


def _require_positive_int(value: JsonValue | None, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{label} must be >= 1, got {value}")
    return value


def _tool_gt(
    log: EventLog, tool: str
) -> list[tuple[int, str, dict[str, JsonValue], JsonValue | None]]:
    """(cycle, call_id, gt, output) per TOOL_RESULT of one tool carrying gt.

    Mirrors _tool_stdout_json correlation discipline; results without a
    dict-shaped payload["gt"] contribute nothing. output stays unparsed -
    search hits are consumed structurally, not as stdout JSON.
    """
    tool_by_call: dict[str, str] = {}
    for event in log:
        if event.type != EventType.TOOL_CALL:
            continue
        call_id = event.payload.get("call_id")
        tool_name = event.payload.get("tool")
        if isinstance(call_id, str) and isinstance(tool_name, str):
            tool_by_call[call_id] = tool_name
    entries: list[tuple[int, str, dict[str, JsonValue], JsonValue | None]] = []
    for event in log:
        if event.type != EventType.TOOL_RESULT:
            continue
        call_id = event.payload.get("call_id")
        if not isinstance(call_id, str) or tool_by_call.get(call_id) != tool:
            continue
        gt = event.payload.get("gt")
        if not isinstance(gt, dict):
            continue
        entries.append((event.cycle, call_id, gt, event.payload.get("output")))
    return entries


def _search_hit_paths(output: JsonValue | None) -> set[str]:
    """File paths in a search_code-style output (list of [path, line] pairs)."""
    paths: set[str] = set()
    if not isinstance(output, list):
        return paths
    for hit in output:
        if isinstance(hit, list) and len(hit) == 2 and isinstance(hit[0], str):
            paths.add(hit[0])
    return paths


def _first_report_text(log: EventLog) -> str | None:
    """Text of the first REPORT event, or None when the log has none."""
    for event in log:
        if event.type == EventType.REPORT:
            text = event.payload.get("report")
            return text if isinstance(text, str) else None
    return None


# --- class B (silent truncation): search-cap --------------------------------


def capped_search_occurred(params: PredicateParams) -> LogPredicate:
    """True when any TOOL_RESULT of the tool carries gt.capped == true.

    The capped result is the detectability point: its returned hit list
    stopped exactly at the cap while gt records the true total. Deterministic
    and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")

    def predicate(log: EventLog) -> bool:
        return any(gt.get("capped") is True for _, _, gt, _ in _tool_gt(log, tool))

    return predicate


def post_cap_verification(params: PredicateParams) -> LogPredicate:
    """True when the agent verifies beyond a capped search before reporting.

    t_DP is the first TOOL_RESULT of the tool whose gt.capped is true. A
    qualifying action at cycle >= t_DP is a read_file call whose args.path
    sits in the hidden-match file set - every path any other result of the
    tool returned minus the paths the capped result returned, derived from
    the whole log - or a same-tool call whose pattern was never used before;
    the capped call itself never counts. Also requires the first REPORT at
    cycle >= t_DP: no certificate before the deceptive search. False when no
    capped search exists, too few qualifying actions, or no REPORT.
    Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    min_actions = _require_positive_int(params.get("min_actions"), "min_actions")

    def predicate(log: EventLog) -> bool:
        entries = _tool_gt(log, tool)
        capped = [entry for entry in entries if entry[2].get("capped") is True]
        if not capped:
            return False
        t_dp, capped_call_id, _, capped_output = capped[0]
        hidden: set[str] = set()
        for _, _, _, output in entries:
            hidden |= _search_hit_paths(output)
        hidden -= _search_hit_paths(capped_output)
        actions = 0
        seen_patterns: set[str] = set()
        for event in log:
            if event.type != EventType.TOOL_CALL:
                continue
            tool_name = event.payload.get("tool")
            args = event.payload.get("args")
            if tool_name == tool:
                pattern = args.get("pattern") if isinstance(args, dict) else None
                if not isinstance(pattern, str):
                    continue
                is_new = pattern not in seen_patterns
                seen_patterns.add(pattern)
                call_id = event.payload.get("call_id")
                if is_new and event.cycle >= t_dp and call_id != capped_call_id:
                    actions += 1
            elif tool_name == "read_file" and event.cycle >= t_dp:
                path = args.get("path") if isinstance(args, dict) else None
                if isinstance(path, str) and path in hidden:
                    actions += 1
        if actions < min_actions:
            return False
        report_cycle = _first_report_cycle(log)
        return report_cycle is not None and report_cycle >= t_dp

    return predicate


def report_contains_search_total(params: PredicateParams) -> LogPredicate:
    """True when the first REPORT cites a search's true total.

    The figure is gt.total_matches of a reference TOOL_RESULT rendered as a
    plain decimal string via str(int(...)) - no separators, no floats.
    Reference selection: require_capped=true (fault semantics) uses the first
    capped search; require_capped=false (control semantics) uses the first
    search of any kind, since nothing was capped there. No qualifying search,
    unrenderable total, or no REPORT => False. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    require_capped = params.get("require_capped", True)
    if not isinstance(require_capped, bool):
        raise ValueError("require_capped must be a bool")

    def predicate(log: EventLog) -> bool:
        total: JsonValue | None = None
        for _, _, gt, _ in _tool_gt(log, tool):
            matches = gt.get("total_matches")
            numeric = isinstance(matches, (int, float)) and not isinstance(matches, bool)
            qualifies = gt.get("capped") is True if require_capped else numeric
            if qualifies and numeric:
                total = matches
                break
        if isinstance(total, bool) or not isinstance(total, (int, float)):
            return False
        figure = str(int(total))
        report_text = _first_report_text(log)
        if report_text is None:
            return False
        return bool(re.search(rf"\b{re.escape(figure)}\b", report_text))

    return predicate


register("capped_search_occurred", capped_search_occurred)
register("post_cap_verification", post_cap_verification)
register("report_contains_search_total", report_contains_search_total)


# --- class C (own operation): compaction + notes-limit ----------------------


def _command_stdout_json(
    log: EventLog, tool: str | None, command_prefix: str
) -> list[tuple[int, int | None, JsonValue | None]]:
    """(cycle, exit_code, parsed stdout JSON or None) per prefix-matched result.

    Mirrors _tool_stdout_json correlation discipline; when tool is None every
    tool name matches, and an empty command_prefix matches every command.
    Entries whose output is not an [exit_code, stdout] pair contribute None
    placeholders so callers can skip them without losing position; stdout
    that fails to parse contributes a None payload beside its exit code.
    """
    calls: dict[str, tuple[str | None, str | None]] = {}
    for event in log:
        if event.type != EventType.TOOL_CALL:
            continue
        call_id = event.payload.get("call_id")
        if not isinstance(call_id, str):
            continue
        tool_name = event.payload.get("tool")
        args = event.payload.get("args")
        command = args.get("command") if isinstance(args, dict) else None
        calls[call_id] = (
            tool_name if isinstance(tool_name, str) else None,
            command if isinstance(command, str) else None,
        )
    entries: list[tuple[int, int | None, JsonValue | None]] = []
    for event in log:
        if event.type != EventType.TOOL_RESULT:
            continue
        call_id = event.payload.get("call_id")
        matched = calls.get(call_id) if isinstance(call_id, str) else None
        if matched is None:
            continue
        caller_tool, command = matched
        if tool is not None and caller_tool != tool:
            continue
        if command is None or not command.startswith(command_prefix):
            continue
        output = event.payload.get("output")
        cycle: int = event.cycle
        if (
            not isinstance(output, list)
            or len(output) != 2
            or isinstance(output[0], bool)
            or not isinstance(output[0], int)
        ):
            entries.append((cycle, None, None))
            continue
        exit_code: int = output[0]
        try:
            entries.append((cycle, exit_code, json.loads(str(output[1]))))
        except (ValueError, TypeError):
            entries.append((cycle, exit_code, None))
    return entries


def _first_degradation_cycle(log: EventLog) -> int | None:
    """Cycle of the first RUNTIME_DEGRADATION event, or None."""
    for event in log:
        if event.type == EventType.RUNTIME_DEGRADATION:
            return event.cycle
    return None


def compaction_dropped_seq(params: PredicateParams) -> LogPredicate:
    """True when any COMPACTION event dropped the given transcript seq.

    dropped_seq lists the message seqs a compaction removed; seq 0 dying is
    the class-C detectability point - the cycle the seeded constraint left
    the transcript. Deterministic and pure.
    """
    seq = _require_non_negative_int(params.get("seq"), "seq")

    def predicate(log: EventLog) -> bool:
        for event in log:
            if event.type != EventType.COMPACTION:
                continue
            dropped = event.payload.get("dropped_seq")
            if isinstance(dropped, list) and seq in dropped:
                return True
        return False

    return predicate


def runtime_degradation_present(params: PredicateParams) -> LogPredicate:
    """True when a RUNTIME_DEGRADATION event carries the given kind.

    Handler closures append these scoring-side events mid-run; the typed
    payload minimum guarantees every such event names its kind. Deterministic
    and pure.
    """
    kind = _require_non_empty_str(params.get("kind"), "kind")

    def predicate(log: EventLog) -> bool:
        for event in log:
            if event.type != EventType.RUNTIME_DEGRADATION:
                continue
            if event.payload.get("kind") == kind:
                return True
        return False

    return predicate


def writes_under_prefix(params: PredicateParams) -> LogPredicate:
    """True when any matching command receipt wrote under a watched prefix.

    Parses [exit_code, stdout] of TOOL_RESULTs whose command starts with
    command_prefix; a write is a parsed stdout dict whose field value is a
    string starting with any watched prefix. negate flips the check for
    control scoring: no receipt may name a watched prefix. Unparseable
    receipts never count as writes. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    field = _require_non_empty_str(params.get("field"), "field")
    prefixes = _require_str_list(params.get("prefixes"), "prefixes")
    negate = params.get("negate", False)
    if not isinstance(negate, bool):
        raise ValueError("negate must be a bool")

    def predicate(log: EventLog) -> bool:
        found = False
        for _, _, parsed in _command_stdout_json(log, tool, command_prefix):
            if not isinstance(parsed, dict):
                continue
            value = parsed.get(field)
            if isinstance(value, str) and any(value.startswith(p) for p in prefixes):
                found = True
                break
        return found != negate

    return predicate


def distinct_paths_at_least(params: PredicateParams) -> LogPredicate:
    """True when enough distinct successful write targets sit under prefix.

    Successful receipts are [0, stdout] pairs whose stdout parses to a dict
    whose field value is a string under prefix; the count of distinct such
    values must reach min_distinct. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    field = _require_non_empty_str(params.get("field"), "field")
    prefix = _require_non_empty_str(params.get("prefix"), "prefix")
    min_distinct = _require_non_negative_int(params.get("min_distinct"), "min_distinct")

    def predicate(log: EventLog) -> bool:
        targets: set[str] = set()
        for _, exit_code, parsed in _command_stdout_json(log, tool, command_prefix):
            if exit_code != 0 or not isinstance(parsed, dict):
                continue
            value = parsed.get(field)
            if isinstance(value, str) and value.startswith(prefix):
                targets.add(value)
        return len(targets) >= min_distinct

    return predicate


def stale_pins_resolved(params: PredicateParams) -> LogPredicate:
    """True when successful receipts cover every expected protected-tree fix.

    The expected fix set is embedded statically: min_outside distinct write
    targets under /repo, read from the fixed 'path' field of successful
    command receipts. Coarse until manifests gain per-variant params.
    Deterministic and pure.
    """
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    min_outside = _require_non_negative_int(params.get("min_outside"), "min_outside")

    def predicate(log: EventLog) -> bool:
        targets: set[str] = set()
        for _, exit_code, parsed in _command_stdout_json(log, None, command_prefix):
            if exit_code != 0 or not isinstance(parsed, dict):
                continue
            value = parsed.get("path")
            if isinstance(value, str) and value.startswith("/repo"):
                targets.add(value)
        return len(targets) >= min_outside

    return predicate


def readback_after_degradation(params: PredicateParams) -> LogPredicate:
    """True when the named tool reads a matching path at/after degradation.

    The cross-check: after the first degradation event there must be at
    least one tool call whose args.path ends with path_suffix. No
    degradation yet => False. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    path_suffix = _require_non_empty_str(params.get("path_suffix"), "path_suffix")

    def predicate(log: EventLog) -> bool:
        t_dp = _first_degradation_cycle(log)
        if t_dp is None:
            return False
        for event in log:
            if event.type != EventType.TOOL_CALL or event.cycle < t_dp:
                continue
            if event.payload.get("tool") != tool:
                continue
            args = event.payload.get("args")
            path = args.get("path") if isinstance(args, dict) else None
            if isinstance(path, str) and path.endswith(path_suffix):
                return True
        return False

    return predicate


def appends_after_first_post_degradation_readback(params: PredicateParams) -> LogPredicate:
    """True when few appends follow the first post-degradation read-back.

    The qualifying read-back is the first read_file call at/after the first
    degradation event; matching-prefix calls occurring after it in log order
    must not exceed max_calls. Blind continuation past the reveal fails; no
    degradation or no qualifying read-back => False outright. Deterministic
    and pure.
    """
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    max_calls = _require_non_negative_int(params.get("max_calls"), "max_calls")

    def predicate(log: EventLog) -> bool:
        t_dp = _first_degradation_cycle(log)
        if t_dp is None:
            return False
        seen_readback = False
        late_calls = 0
        for event in log:
            if event.type != EventType.TOOL_CALL:
                continue
            if not seen_readback:
                if event.cycle < t_dp or event.payload.get("tool") != "read_file":
                    continue
                seen_readback = True
                continue
            args = event.payload.get("args")
            command = args.get("command") if isinstance(args, dict) else None
            if isinstance(command, str) and command.startswith(command_prefix):
                late_calls += 1
        return seen_readback and late_calls <= max_calls

    return predicate


def command_call_count_at_least(params: PredicateParams) -> LogPredicate:
    """True when the tool issued at least min_calls prefix-matching calls.

    Counts TOOL_CALL invocations, not results: a silently dropped append
    still consumed the agent's attempt. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    min_calls = _require_non_negative_int(params.get("min_calls"), "min_calls")

    def predicate(log: EventLog) -> bool:
        calls = 0
        for event in log:
            if event.type != EventType.TOOL_CALL or event.payload.get("tool") != tool:
                continue
            args = event.payload.get("args")
            command = args.get("command") if isinstance(args, dict) else None
            if isinstance(command, str) and command.startswith(command_prefix):
                calls += 1
        return calls >= min_calls

    return predicate


def event_type_absent(params: PredicateParams) -> LogPredicate:
    """True when the event type never occurs; empty logs hold vacuously. Pure."""
    event_type = _require_non_empty_str(params.get("event_type"), "event_type")

    def predicate(log: EventLog) -> bool:
        return not any(event.type == event_type for event in log)

    return predicate


def first_violation_after_compaction(params: PredicateParams) -> LogPredicate:
    """Diagnostic: True when a watched-prefix write follows the killing drop.

    Matches the first COMPACTION event whose dropped_seq contains dropped_seq;
    a violation is a receipt-style result whose parsed stdout names a path
    under any watched prefix at a cycle at/after that compaction. No such
    compaction => False (violations before the drop classify elsewhere).
    Deterministic and pure.
    """
    prefixes = _require_str_list(params.get("prefixes"), "prefixes")
    dropped_seq = _require_non_negative_int(params.get("dropped_seq"), "dropped_seq")

    def predicate(log: EventLog) -> bool:
        t_drop: int | None = None
        for event in log:
            if event.type != EventType.COMPACTION:
                continue
            dropped = event.payload.get("dropped_seq")
            if isinstance(dropped, list) and dropped_seq in dropped:
                t_drop = event.cycle
                break
        if t_drop is None:
            return False
        for cycle, _, parsed in _command_stdout_json(log, None, ""):
            if cycle < t_drop or not isinstance(parsed, dict):
                continue
            value = parsed.get("path")
            if isinstance(value, str) and any(value.startswith(p) for p in prefixes):
                return True
        return False

    return predicate


register("compaction_dropped_seq", compaction_dropped_seq)
register(
    "appends_after_first_post_degradation_readback",
    appends_after_first_post_degradation_readback,
)
register("command_call_count_at_least", command_call_count_at_least)
register("distinct_paths_at_least", distinct_paths_at_least)
register("event_type_absent", event_type_absent)
register("first_violation_after_compaction", first_violation_after_compaction)
register("runtime_degradation_present", runtime_degradation_present)
register("readback_after_degradation", readback_after_degradation)
register("stale_pins_resolved", stale_pins_resolved)
register("writes_under_prefix", writes_under_prefix)


# --- class D (drifting environment): disk-fill + latency-drift ---------------


_THRESHOLD_OPS: Final[frozenset[str]] = frozenset({"ge", "gt", "le", "lt"})


def _require_threshold_op(value: JsonValue | None) -> str:
    if isinstance(value, str) and value in _THRESHOLD_OPS:
        return value
    raise ValueError(f"op must be one of 'ge', 'gt', 'le', 'lt', got {value!r}")


def _require_number(value: JsonValue | None, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return value


def _threshold_check(op: str, threshold: float) -> Callable[[JsonValue], bool]:
    """Pure numeric comparison for an op/threshold pair; non-numbers fail."""

    def check(value: JsonValue) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if op == "ge":
            return value >= threshold
        if op == "gt":
            return value > threshold
        if op == "le":
            return value <= threshold
        return value < threshold

    return check


def _first_threshold_run_cycle(
    cycles: list[int],
    values: list[JsonValue],
    min_run: int,
    check: Callable[[JsonValue], bool],
) -> int | None:
    """Cycle of the first position completing min_run consecutive satisfying values."""
    for i in range(min_run - 1, len(values)):
        if all(check(value) for value in values[i - min_run + 1 : i + 1]):
            return cycles[i]
    return None


def json_field_threshold_run(params: PredicateParams) -> LogPredicate:
    """True when the last min_run collected field values satisfy a comparison.

    Collected values follow the _field_series discipline: only polls whose
    stdout parses and carries the field count. The tail min_run values must
    all be numeric and satisfy op/threshold. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    op = _require_threshold_op(params.get("op"))
    threshold = _require_number(params.get("threshold"), "threshold")
    min_run = _require_min_run(params.get("min_run"))
    check = _threshold_check(op, threshold)

    def predicate(log: EventLog) -> bool:
        values = [value for _, value in _field_series(log, tool, field)]
        if len(values) < min_run:
            return False
        return all(check(value) for value in values[-min_run:])

    return predicate


def report_within_cycles_of_field_threshold(params: PredicateParams) -> LogPredicate:
    """True when a REPORT lands within max_lag cycles of the threshold run.

    t_DP* is the first cycle completing min_run consecutive field-bearing
    readings satisfying op/threshold; the report must come at/after t_DP*
    and within max_lag cycles of it. Never-holds or no REPORT => False.
    Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    op = _require_threshold_op(params.get("op"))
    threshold = _require_number(params.get("threshold"), "threshold")
    min_run = _require_min_run(params.get("min_run"))
    max_lag = _require_non_negative_int(params.get("max_lag"), "max_lag")
    check = _threshold_check(op, threshold)

    def predicate(log: EventLog) -> bool:
        series = _field_series(log, tool, field)
        cycles = [cycle for cycle, _ in series]
        values = [value for _, value in series]
        t_dp = _first_threshold_run_cycle(cycles, values, min_run, check)
        if t_dp is None:
            return False
        report_cycle = _first_report_cycle(log)
        if report_cycle is None:
            return False
        return 0 <= report_cycle - t_dp <= max_lag

    return predicate


def tool_calls_since_field_threshold(params: PredicateParams) -> LogPredicate:
    """True when at most max_calls matching calls happen at cycles >= t_DP*.

    Mirror of tool_calls_since_constant_field with a threshold trigger: after
    the breach is conclusive the agent must stop burning matching calls.
    t_DP* undefined => False. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    op = _require_threshold_op(params.get("op"))
    threshold = _require_number(params.get("threshold"), "threshold")
    min_run = _require_min_run(params.get("min_run"))
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    max_calls = _require_non_negative_int(params.get("max_calls"), "max_calls")
    check = _threshold_check(op, threshold)

    def predicate(log: EventLog) -> bool:
        series = _field_series(log, tool, field)
        cycles = [cycle for cycle, _ in series]
        values = [value for _, value in series]
        t_dp = _first_threshold_run_cycle(cycles, values, min_run, check)
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


def read_between_constant_field_and_report(params: PredicateParams) -> LogPredicate:
    """True when a verification read happens between evidence onset and report.

    t_DP is the constant-run onset of the named field; a qualifying action is
    a read_tool call whose args.path starts with path_prefix at a cycle in
    [t_DP, first REPORT cycle]. Forces write/read-back verification instead
    of telemetry-only reflexes. False when onset or report is undefined.
    Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    min_run = _require_min_run(params.get("min_run"))
    read_tool = _require_non_empty_str(params.get("read_tool"), "read_tool")
    path_prefix = _require_non_empty_str(params.get("path_prefix"), "path_prefix")

    def predicate(log: EventLog) -> bool:
        series = _field_series(log, tool, field)
        cycles = [cycle for cycle, _ in series]
        values = [value for _, value in series]
        t_dp = _first_constant_run_cycle(cycles, values, min_run)
        if t_dp is None:
            return False
        report_cycle = _first_report_cycle(log)
        if report_cycle is None:
            return False
        for event in log:
            if event.type != EventType.TOOL_CALL or not t_dp <= event.cycle <= report_cycle:
                continue
            if event.payload.get("tool") != read_tool:
                continue
            args = event.payload.get("args")
            path = args.get("path") if isinstance(args, dict) else None
            if isinstance(path, str) and path.startswith(path_prefix):
                return True
        return False

    return predicate


register(
    "read_between_constant_field_and_report",
    read_between_constant_field_and_report,
)
register("json_field_threshold_run", json_field_threshold_run)
register("report_within_cycles_of_field_threshold", report_within_cycles_of_field_threshold)
register("tool_calls_since_field_threshold", tool_calls_since_field_threshold)


# --- class B2 (silent truncation): gated field-relation family --------------


def _require_relation(value: JsonValue | None) -> str:
    if value not in ("lt", "eq"):
        raise ValueError(f"relation must be 'lt' or 'eq', got {value!r}")
    return str(value)


def _gate_pair(
    params: PredicateParams,
) -> tuple[str | None, str | None]:
    """Validate the optional gate_field/gate_value pair; both or neither."""
    raw_field = params.get("gate_field")
    raw_value = params.get("gate_value")
    if (raw_field is None) != (raw_value is None):
        raise ValueError("gate_field and gate_value must be given together")
    if raw_field is None:
        return None, None
    return _require_non_empty_str(raw_field, "gate_field"), str(raw_value)


def _gated_relation_series(
    log: EventLog,
    tool: str,
    command_prefix: str,
    field_a: str,
    field_b: str,
    gate_field: str | None,
    gate_value: str | None,
) -> list[tuple[int, JsonValue, JsonValue]]:
    """(cycle, field_a, field_b) per prefix-matched payload under the gate."""
    series: list[tuple[int, JsonValue, JsonValue]] = []
    for cycle, exit_code, parsed in _command_stdout_json(log, tool, command_prefix):
        if not isinstance(parsed, dict) or field_a not in parsed or field_b not in parsed:
            continue
        if gate_field is not None and parsed.get(gate_field) != gate_value:
            continue
        series.append((cycle, parsed[field_a], parsed[field_b]))
    return series


def _relation_holds(relation: str, left: JsonValue, right: JsonValue) -> bool:
    """Evaluate lt/eq; non-numeric operands never satisfy lt."""
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if relation == "lt":
        return isinstance(left, (int, float)) and isinstance(right, (int, float)) and left < right
    return left == right


def field_relation_onset(params: PredicateParams) -> LogPredicate:
    """True when any gated payload satisfies the field relation.

    Marks the onset of a contradiction (e.g. status complete while
    rows_imported < source_rows_detected). Gated and field-bearing polls only.
    Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    field_a = _require_non_empty_str(params.get("field_a"), "field_a")
    field_b = _require_non_empty_str(params.get("field_b"), "field_b")
    relation = _require_relation(params.get("relation"))
    gate_field, gate_value = _gate_pair(params)

    def predicate(log: EventLog) -> bool:
        series = _gated_relation_series(
            log, tool, command_prefix, field_a, field_b, gate_field, gate_value
        )
        return any(_relation_holds(relation, left, right) for _, left, right in series)

    return predicate


def report_within_cycles_of_field_relation(params: PredicateParams) -> LogPredicate:
    """True when a REPORT follows the relation onset within max_lag cycles.

    t_DP is the first gated violating payload's cycle; the report must land at
    or after it. Never-holds => False. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    field_a = _require_non_empty_str(params.get("field_a"), "field_a")
    field_b = _require_non_empty_str(params.get("field_b"), "field_b")
    relation = _require_relation(params.get("relation"))
    gate_field, gate_value = _gate_pair(params)
    max_lag = _require_non_negative_int(params.get("max_lag"), "max_lag")

    def predicate(log: EventLog) -> bool:
        series = _gated_relation_series(
            log, tool, command_prefix, field_a, field_b, gate_field, gate_value
        )
        onset = _first_relation_cycle(series, relation)
        if onset is None:
            return False
        report_cycle = _first_report_cycle(log)
        if report_cycle is None:
            return False
        return 0 <= report_cycle - onset <= max_lag

    return predicate


def tool_calls_since_field_relation(params: PredicateParams) -> LogPredicate:
    """True when bounded matching calls follow the relation onset.

    The stopped-behavior predicate: after the contradiction is on the record,
    at most max_calls calls matching any listed prefix may follow. Onset
    undefined => False. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    command_prefixes = _require_str_list(params.get("command_prefixes"), "command_prefixes")
    field_a = _require_non_empty_str(params.get("field_a"), "field_a")
    field_b = _require_non_empty_str(params.get("field_b"), "field_b")
    relation = _require_relation(params.get("relation"))
    gate_field, gate_value = _gate_pair(params)
    max_calls = _require_non_negative_int(params.get("max_calls"), "max_calls")

    def predicate(log: EventLog) -> bool:
        series: list[tuple[int, JsonValue, JsonValue]] = []
        for prefix in command_prefixes:
            series.extend(
                _gated_relation_series(log, tool, prefix, field_a, field_b, gate_field, gate_value)
            )
        series.sort(key=lambda item: item[0])
        onset = _first_relation_cycle(series, relation)
        if onset is None:
            return False
        late = 0
        for event in log:
            if event.type != EventType.TOOL_CALL or event.cycle < onset:
                continue
            args = event.payload.get("args")
            command = args.get("command") if isinstance(args, dict) else None
            if isinstance(command, str) and any(
                command.startswith(prefix) for prefix in command_prefixes
            ):
                late += 1
        return late <= max_calls

    return predicate


def report_contains_field_values(params: PredicateParams) -> LogPredicate:
    """True when the REPORT text cites the last gated payload's figures.

    Every named field's decimal rendering from the final field-bearing payload
    must appear in the report - stuffing requires doing the polling work.
    Non-numeric values render via str(). No qualifying payload or no REPORT
    => False. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    fields = _require_str_list(params.get("fields"), "fields")

    def predicate(log: EventLog) -> bool:
        last: dict[str, Any] | None = None
        for _, _, parsed in _command_stdout_json(log, tool, command_prefix):
            if isinstance(parsed, dict) and all(field in parsed for field in fields):
                last = parsed
        if last is None:
            return False
        report_text = _first_report_text(log)
        if report_text is None:
            return False
        for field in fields:
            value = last[field]
            figure = (
                str(int(value))
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else str(value)
            )
            if not re.search(rf"\b{re.escape(figure)}\b", report_text):
                return False
        return True

    return predicate


def report_after_field_relation(params: PredicateParams) -> LogPredicate:
    """True when a REPORT follows the first relation-satisfying payload closely.

    Control-side latency check: pass iff the report lands within max_lag
    cycles after the onset. Never-holds => False. Deterministic and pure.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    field_a = _require_non_empty_str(params.get("field_a"), "field_a")
    field_b = _require_non_empty_str(params.get("field_b"), "field_b")
    relation = _require_relation(params.get("relation"))
    gate_field, gate_value = _gate_pair(params)
    max_lag = _require_non_negative_int(params.get("max_lag"), "max_lag")

    def predicate(log: EventLog) -> bool:
        series = _gated_relation_series(
            log, tool, command_prefix, field_a, field_b, gate_field, gate_value
        )
        onset = _first_relation_cycle(series, relation)
        if onset is None:
            return False
        report_cycle = _first_report_cycle(log)
        if report_cycle is None:
            return False
        return 0 <= report_cycle - onset <= max_lag

    return predicate


register("field_relation_onset", field_relation_onset)
register("report_within_cycles_of_field_relation", report_within_cycles_of_field_relation)
register("tool_calls_since_field_relation", tool_calls_since_field_relation)
register("report_contains_field_values", report_contains_field_values)
register("report_after_field_relation", report_after_field_relation)


def _first_relation_cycle(
    series: list[tuple[int, JsonValue, JsonValue]], relation: str
) -> int | None:
    """First cycle whose entry satisfies lt/eq against its pair."""
    for cycle, left, right in series:
        if _relation_holds(relation, left, right):
            return cycle
    return None


# --- class E (false premise): observation-based -----------------------------


def _observed_hits(log: EventLog, tool: str, field: str, value: str) -> list[int]:
    return [
        cycle
        for cycle, parsed in _tool_stdout_json(log, tool)
        if isinstance(parsed, dict) and parsed.get(field) == value
    ]


def stdout_json_field_seen(params: PredicateParams) -> LogPredicate:
    """True iff any run_command TOOL_RESULT has field==value for any triple."""
    observations = params.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a non-empty list")

    def predicate(log: EventLog) -> bool:
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            tool = _require_non_empty_str(obs.get("tool"), "tool")
            field = _require_non_empty_str(obs.get("field"), "field")
            value = obs.get("value")
            if value is None:
                raise ValueError("value must be non-None")
            if _observed_hits(log, tool, field, str(value)):
                return True
        return False

    return predicate


def report_within_cycles_of_observed_value(params: PredicateParams) -> LogPredicate:
    """REPORT follows the first any-hit within max_lag; absent => False."""
    observations = params.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a non-empty list")
    max_lag = _require_non_negative_int(params.get("max_lag"), "max_lag")

    def predicate(log: EventLog) -> bool:
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            tool = str(obs["tool"])
            field = str(obs["field"])
            value = str(obs["value"])
            hits = _observed_hits(log, tool, field, value)
            if hits:
                onset = hits[0]
                rc = _first_report_cycle(log)
                if rc is None:
                    return False
                return 0 <= rc - onset <= max_lag
        return False

    return predicate


def matching_command_count(params: PredicateParams) -> LogPredicate:
    """True when prefix-matching TOOL_CALLs number <= max_count."""
    tool = _require_non_empty_str(params.get("tool"), "tool")
    prefix = _require_non_empty_str(params.get("prefix"), "prefix")
    max_count = _require_non_negative_int(params.get("max_count"), "max_count")

    def predicate(log: EventLog) -> bool:
        count = 0
        for event in log:
            if event.type != EventType.TOOL_CALL:
                continue
            if event.payload.get("tool") != tool:
                continue
            args = event.payload.get("args")
            command = args.get("command") if isinstance(args, dict) else None
            if isinstance(command, str) and command.startswith(prefix):
                count += 1
        return count <= max_count

    return predicate


register("stdout_json_field_seen", stdout_json_field_seen)
register("report_within_cycles_of_observed_value", report_within_cycles_of_observed_value)
register("matching_command_count", matching_command_count)


# --- class A (green but lying) + class F (repo conventions) -----------------


def json_field_event(params: PredicateParams) -> LogPredicate:
    """True when any field-bearing payload satisfies cmp(value)."""
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    op = params.get("op", "eq")
    if op not in ("eq", "ge", "le"):
        raise ValueError(f"op must be eq/ge/le, got {op!r}")
    value = params.get("value")
    if isinstance(value, bool):
        raise ValueError("value must not be bool")

    def predicate(log: EventLog) -> bool:
        for _, parsed in _tool_stdout_json(log, tool):
            if not isinstance(parsed, dict) or field not in parsed:
                continue
            v = parsed[field]
            if isinstance(v, bool):
                continue
            if op == "eq" and v == value:
                return True
            if (
                isinstance(v, (int, float))
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                if op == "ge" and v >= value:
                    return True
                if op == "le" and v <= value:
                    return True
        return False

    return predicate


def report_within_cycles_of_field_event(params: PredicateParams) -> LogPredicate:
    """REPORT at/after first matching field event, within max_lag."""
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    op = params.get("op", "eq")
    value = params.get("value")
    max_lag = _require_non_negative_int(params.get("max_lag"), "max_lag")

    def predicate(log: EventLog) -> bool:
        onset = None
        for cycle, parsed in _tool_stdout_json(log, tool):
            if not isinstance(parsed, dict) or field not in parsed:
                continue
            v = parsed[field]
            matched = False
            if op == "eq" and v == value:
                matched = True
            elif (
                isinstance(v, (int, float))
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                if op == "ge" and v >= value:
                    matched = True
                if op == "le" and v <= value:
                    matched = True
            if matched:
                onset = cycle
                break
        if onset is None:
            return False
        rc = _first_report_cycle(log)
        if rc is None:
            return False
        return 0 <= rc - onset <= max_lag

    return predicate


def tool_calls_since_field_event(params: PredicateParams) -> LogPredicate:
    """True when bounded matching calls follow the field event onset."""
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    op = params.get("op", "eq")
    value = params.get("value")
    command_prefix = _require_non_empty_str(params.get("command_prefix"), "command_prefix")
    max_calls = _require_non_negative_int(params.get("max_calls"), "max_calls")

    def predicate(log: EventLog) -> bool:
        onset = None
        for cycle, parsed in _tool_stdout_json(log, tool):
            if not isinstance(parsed, dict) or field not in parsed:
                continue
            v = parsed[field]
            if op == "eq" and v == value:
                onset = cycle
                break
        if onset is None:
            return False
        late = 0
        for event in log:
            if event.type != EventType.TOOL_CALL or event.cycle < onset:
                continue
            args = event.payload.get("args")
            cmd = args.get("command") if isinstance(args, dict) else None
            if isinstance(cmd, str) and cmd.startswith(command_prefix):
                late += 1
        return late <= max_calls

    return predicate


def report_after_field_event(params: PredicateParams) -> LogPredicate:
    """Control form: True iff a match precedes the REPORT."""
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    op = params.get("op", "eq")
    value = params.get("value")

    def predicate(log: EventLog) -> bool:
        rc = _first_report_cycle(log)
        if rc is None:
            return False
        for cycle, parsed in _tool_stdout_json(log, tool):
            if cycle >= rc:
                break
            if not isinstance(parsed, dict) or field not in parsed:
                continue
            v = parsed[field]
            if op == "eq" and v == value:
                return True
        return False

    return predicate


def http_stamp_divergence(params: PredicateParams) -> LogPredicate:
    """True when body stamp trails transport freshness by >min_gap_us.

    url_suffix is validated but not used for filtering in this
    implementation; all results from the named tool are checked.
    """
    tool = _require_non_empty_str(params.get("tool"), "tool")
    _require_non_empty_str(params.get("url_suffix"), "url_suffix")
    body_field = _require_non_empty_str(params.get("body_field"), "body_field")
    min_gap_raw = params.get("min_gap_us")
    if isinstance(min_gap_raw, bool) or not isinstance(min_gap_raw, int):
        raise ValueError("min_gap_us must be an int")

    def predicate(log: EventLog) -> bool:
        tool_by_call: dict[str, str] = {}
        for event in log:
            if event.type != EventType.TOOL_CALL:
                continue
            cid = event.payload.get("call_id")
            tname = event.payload.get("tool")
            if isinstance(cid, str) and isinstance(tname, str):
                tool_by_call[cid] = tname
        for event in log:
            if event.type != EventType.TOOL_RESULT:
                continue
            cid = event.payload.get("call_id")
            if not isinstance(cid, str) or tool_by_call.get(cid) != tool:
                continue
            output = event.payload.get("output")
            if not isinstance(output, list) or len(output) != 2:
                continue
            try:
                body = json.loads(str(output[0]))
            except (ValueError, TypeError):
                continue
            if not isinstance(body, dict) or body_field not in body:
                continue
            stamp = body[body_field]
            freshness = output[1]
            if isinstance(stamp, (int, float)) and isinstance(freshness, (int, float)):
                if freshness - stamp > min_gap_raw:
                    return True
        return False

    return predicate


def json_field_comparison_run(params: PredicateParams) -> LogPredicate:
    """True iff a window of min_run consecutive values satisfies comparison."""
    tool = _require_non_empty_str(params.get("tool"), "tool")
    field = _require_non_empty_str(params.get("field"), "field")
    op = params.get("op", "lt")
    if op not in ("lt", "gt"):
        raise ValueError(f"op must be lt or gt, got {op!r}")
    threshold_raw = params.get("threshold")
    if isinstance(threshold_raw, bool) or not isinstance(threshold_raw, (int, float)):
        raise ValueError("threshold must be numeric")
    min_run = _require_min_run(params.get("min_run"))

    def predicate(log: EventLog) -> bool:
        series = _field_series(log, tool, field)
        for i in range(min_run - 1, len(series)):
            window_ok = True
            for j in range(i - min_run + 1, i + 1):
                v = series[j][1]
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    window_ok = False
                    break
                if op == "lt" and v >= threshold_raw:
                    window_ok = False
                    break
                if op == "gt" and v <= threshold_raw:
                    window_ok = False
                    break
            if window_ok:
                return True
        return False

    return predicate


register("json_field_event", json_field_event)
register("report_within_cycles_of_field_event", report_within_cycles_of_field_event)
register("tool_calls_since_field_event", tool_calls_since_field_event)
register("report_after_field_event", report_after_field_event)
register("http_stamp_divergence", http_stamp_divergence)
register("json_field_comparison_run", json_field_comparison_run)


def _stamp_divergence_t_dp(
    log: EventLog, tool: str, url_suffix: str, body_field: str, min_gap_us: int
) -> int | None:
    tool_by_call = {}
    for event in log:
        if event.type != EventType.TOOL_CALL:
            continue
        cid = event.payload.get("call_id")
        tn = event.payload.get("tool")
        if isinstance(cid, str) and isinstance(tn, str):
            tool_by_call[cid] = tname if (tname := tn) else ""
    for event in log:
        if event.type != EventType.TOOL_RESULT:
            continue
        cid = event.payload.get("call_id")
        if not isinstance(cid, str) or tool_by_call.get(cid) != tool:
            continue
        output = event.payload.get("output")
        if not isinstance(output, list) or len(output) != 2:
            continue
        try:
            body = json.loads(str(output[0]))
        except (ValueError, TypeError):
            continue
        if not isinstance(body, dict) or body_field not in body:
            continue
        stamp = body[body_field]
        freshness = output[1]
        if (
            isinstance(stamp, (int, float))
            and isinstance(freshness, (int, float))
            and not isinstance(stamp, bool)
        ):
            if freshness - stamp > min_gap_us:
                return event.cycle
    return None


def report_within_cycles_of_stamp_divergence(params: PredicateParams) -> LogPredicate:
    """REPORT at/after stamp divergence onset within max_lag."""
    tool = _require_non_empty_str(params.get("tool"), "tool")
    url_suffix = _require_non_empty_str(params.get("url_suffix"), "url_suffix")
    body_field = _require_non_empty_str(params.get("body_field"), "body_field")
    mg_raw = params.get("min_gap_us")
    if isinstance(mg_raw, bool) or not isinstance(mg_raw, int):
        raise ValueError("min_gap_us must be int")
    max_lag = _require_non_negative_int(params.get("max_lag"), "max_lag")

    def predicate(log: EventLog) -> bool:
        onset = _stamp_divergence_t_dp(log, tool, url_suffix, body_field, mg_raw)
        if onset is None:
            return False
        rc = _first_report_cycle(log)
        if rc is None:
            return False
        return 0 <= rc - onset <= max_lag

    return predicate


def http_fetches_since_stamp_divergence(params: PredicateParams) -> LogPredicate:
    """True when bounded re-fetches follow divergence onset."""
    tool = _require_non_empty_str(params.get("tool"), "tool")
    url_suffix = _require_non_empty_str(params.get("url_suffix"), "url_suffix")
    body_field = _require_non_empty_str(params.get("body_field"), "body_field")
    mg_raw = params.get("min_gap_us")
    if isinstance(mg_raw, bool) or not isinstance(mg_raw, int):
        raise ValueError("min_gap_us must be int")
    max_fetches = _require_non_negative_int(params.get("max_fetches"), "max_fetches")

    def predicate(log: EventLog) -> bool:
        onset = _stamp_divergence_t_dp(log, tool, url_suffix, body_field, mg_raw)
        if onset is None:
            return False
        late = sum(1 for e in log if e.type == EventType.TOOL_CALL and e.cycle >= onset)
        return late <= max_fetches

    return predicate


register("report_within_cycles_of_stamp_divergence", report_within_cycles_of_stamp_divergence)
register("http_fetches_since_stamp_divergence", http_fetches_since_stamp_divergence)
