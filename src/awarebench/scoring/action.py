"""Detection-relative action predicates and their dedicated registry."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

from awarebench.events import Event, EventLog, EventType, JsonValue
from awarebench.scoring.predicates import REGISTRY, PredicateFactory, PredicateParams


@dataclass(frozen=True)
class ActionContext:
    """Full evidence plus the only events eligible to count as a response.

    Predicates may consult the complete log to understand prerequisites and
    ground truth. They may classify actions only from post_detection_window,
    which starts at the semantic detection event and ends at the inclusive
    detection_cycle + k deadline.
    """

    full_log: EventLog
    post_detection_window: tuple[Event, ...]
    detection_seq: int
    detection_cycle: int
    deadline_cycle: int

    def window_log(self) -> EventLog:
        """Materialize a locally reindexed view for sequence-independent predicates."""
        window = EventLog()
        for event in self.post_detection_window:
            window.append(event.type, event.cycle, event.t_us, event.payload)
        return window


ActionPredicate = Callable[[ActionContext], bool]
ActionPredicateFactory = Callable[[PredicateParams], ActionPredicate]


def build_action_context(
    log: EventLog,
    *,
    detection_seq: int,
    detection_cycle: int,
    action_window_k: int,
) -> ActionContext:
    """Build one inclusive, sequence-preserving action evaluation context."""
    deadline = detection_cycle + action_window_k
    eligible = tuple(
        event for event in log if event.seq >= detection_seq and event.cycle <= deadline
    )
    return ActionContext(
        full_log=log,
        post_detection_window=eligible,
        detection_seq=detection_seq,
        detection_cycle=detection_cycle,
        deadline_cycle=deadline,
    )


def _windowed(factory: PredicateFactory) -> ActionPredicateFactory:
    """Adapt a log predicate so it can inspect only eligible action events."""

    def build(params: PredicateParams) -> ActionPredicate:
        predicate = factory(params)
        return lambda context: predicate(context.window_log())

    return build


def _non_empty_tools(value: object) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("tools must be a non-empty list of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError("tools must be a non-empty list of strings")
    return set(value)


def _non_empty_str(value: JsonValue | None, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


def tool_call_count(params: PredicateParams) -> ActionPredicate:
    """Bound calls to named tools in the post-detection window."""
    tools = _non_empty_tools(params.get("tools"))
    has_min = "min_count" in params
    has_max = "max_count" in params
    if not has_min and not has_max:
        raise ValueError("tool_call_count requires min_count or max_count")
    min_count = _non_negative_int(params.get("min_count"), "min_count") if has_min else 0
    max_count = _non_negative_int(params.get("max_count"), "max_count") if has_max else None
    if max_count is not None and min_count > max_count:
        raise ValueError("min_count must be <= max_count")

    def predicate(context: ActionContext) -> bool:
        count = sum(
            event.type == EventType.TOOL_CALL and event.payload.get("tool") in tools
            for event in context.post_detection_window
        )
        return count >= min_count and (max_count is None or count <= max_count)

    return predicate


def _call_map(events: list[Event]) -> dict[str, Event]:
    calls: dict[str, Event] = {}
    for event in events:
        if event.type != EventType.TOOL_CALL:
            continue
        call_id = event.payload.get("call_id")
        if isinstance(call_id, str):
            calls[call_id] = event
    return calls


def _stdout_json(output: JsonValue | None) -> tuple[int, dict[str, JsonValue]] | None:
    if not isinstance(output, list) or len(output) != 2:
        return None
    exit_code, stdout = output
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or not isinstance(stdout, str):
        return None
    try:
        parsed = json.loads(stdout)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return exit_code, parsed


def _search_paths(output: JsonValue | None) -> set[str]:
    if not isinstance(output, list):
        return set()
    return {
        item[0]
        for item in output
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str)
    }


def _is_grounded_search_refinement(pattern: str, capped_patterns: set[str]) -> bool:
    """Accept only narrow regex forms that retain a prior capped target."""
    return any(
        pattern != capped_pattern
        and pattern
        in {
            f"{capped_pattern}$",
            f"{capped_pattern}\\b",
            f"{capped_pattern}\\(",
            f"{capped_pattern}\\s*\\(",
            f"^{capped_pattern}",
            f"\\b{capped_pattern}\\b",
            f"\\b{capped_pattern}\\(",
        }
        for capped_pattern in capped_patterns
    )


def _index_lists_path(index_text: str, path: str) -> bool:
    """Match either a full inventory path or its basename as emitted by generators."""
    basename = PurePosixPath(path).name
    listed = {
        line.strip().removeprefix("- ").removeprefix("* ").strip("`")
        for line in index_text.splitlines()
    }
    return path in listed or basename in listed


def post_detection_search_verification(params: PredicateParams) -> ActionPredicate:
    """Require a novel, ground-truth-backed verification after detection."""
    tool = _non_empty_str(params.get("tool"), "tool")
    min_actions = _non_negative_int(params.get("min_actions"), "min_actions")
    if min_actions < 1:
        raise ValueError("min_actions must be >= 1")

    def predicate(context: ActionContext) -> bool:
        full_events = [event for event in context.full_log if event.cycle <= context.deadline_cycle]
        calls = _call_map(full_events)
        prior_patterns: set[str] = set()
        capped_patterns: set[str] = set()
        capped_paths: set[str] = set()
        index_texts: list[tuple[int, str]] = []
        for event in full_events:
            if (
                event.type == EventType.TOOL_CALL
                and event.seq < context.detection_seq
                and event.payload.get("tool") == tool
            ):
                args = event.payload.get("args")
                pattern = args.get("pattern") if isinstance(args, dict) else None
                if isinstance(pattern, str):
                    prior_patterns.add(pattern)
            if event.type != EventType.TOOL_RESULT:
                continue
            call_id = event.payload.get("call_id")
            call = calls.get(call_id) if isinstance(call_id, str) else None
            if call is None:
                continue
            gt = event.payload.get("gt")
            if (
                event.seq < context.detection_seq
                and call.payload.get("tool") == tool
                and isinstance(gt, dict)
                and gt.get("capped") is True
            ):
                capped_paths |= _search_paths(event.payload.get("output"))
                args = call.payload.get("args")
                pattern = args.get("pattern") if isinstance(args, dict) else None
                if isinstance(pattern, str):
                    capped_patterns.add(pattern)
            if call.payload.get("tool") == "read_file":
                args = call.payload.get("args")
                path = args.get("path") if isinstance(args, dict) else None
                output = event.payload.get("output")
                if isinstance(path, str) and path.endswith("INDEX.md") and isinstance(output, str):
                    index_texts.append((event.seq, output))

        post_calls = _call_map(list(context.post_detection_window))
        actions = 0
        for event in context.post_detection_window:
            if event.type != EventType.TOOL_RESULT:
                continue
            call_id = event.payload.get("call_id")
            call = post_calls.get(call_id) if isinstance(call_id, str) else None
            if call is None:
                continue
            call_tool = call.payload.get("tool")
            args = call.payload.get("args")
            if call_tool == tool:
                pattern = args.get("pattern") if isinstance(args, dict) else None
                gt = event.payload.get("gt")
                total_matches = gt.get("total_matches") if isinstance(gt, dict) else None
                returned_matches = gt.get("returned_matches") if isinstance(gt, dict) else None
                output_paths = _search_paths(event.payload.get("output"))
                if (
                    isinstance(pattern, str)
                    and pattern not in prior_patterns
                    and _is_grounded_search_refinement(pattern, capped_patterns)
                    and isinstance(gt, dict)
                    and gt.get("capped") is False
                    and isinstance(total_matches, int)
                    and not isinstance(total_matches, bool)
                    and isinstance(returned_matches, int)
                    and not isinstance(returned_matches, bool)
                    and total_matches > 0
                    and returned_matches == total_matches
                    and bool(output_paths - capped_paths)
                ):
                    actions += 1
                    prior_patterns.add(pattern)
            elif call_tool == "read_file":
                path = args.get("path") if isinstance(args, dict) else None
                if (
                    isinstance(path, str)
                    and path not in capped_paths
                    and not path.endswith("INDEX.md")
                    and any(
                        index_seq < call.seq and _index_lists_path(index_text, path)
                        for index_seq, index_text in index_texts
                    )
                ):
                    actions += 1
            if actions >= min_actions:
                return True
        return False

    return predicate


def restage_then_verified_build(params: PredicateParams) -> ActionPredicate:
    """Require successful restage and later build receipts with matching revisions."""
    tool = _non_empty_str(params.get("tool"), "tool")
    restage_prefix = _non_empty_str(params.get("restage_prefix"), "restage_prefix")
    build_prefix = _non_empty_str(params.get("build_prefix"), "build_prefix")
    staged_field = _non_empty_str(params.get("staged_field"), "staged_field")
    embedded_field = _non_empty_str(params.get("embedded_field"), "embedded_field")

    def predicate(context: ActionContext) -> bool:
        calls = _call_map(list(context.post_detection_window))
        staged: list[tuple[int, float]] = []
        built: list[tuple[int, float]] = []
        for event in context.post_detection_window:
            if event.type != EventType.TOOL_RESULT:
                continue
            call_id = event.payload.get("call_id")
            call = calls.get(call_id) if isinstance(call_id, str) else None
            if call is None or call.payload.get("tool") != tool:
                continue
            args = call.payload.get("args")
            command = args.get("command") if isinstance(args, dict) else None
            receipt = _stdout_json(event.payload.get("output"))
            if not isinstance(command, str) or receipt is None or receipt[0] != 0:
                continue
            parsed = receipt[1]
            if command.startswith(restage_prefix):
                value = parsed.get(staged_field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    staged.append((event.seq, float(value)))
            elif command.startswith(build_prefix):
                value = parsed.get(embedded_field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    built.append((event.seq, float(value)))
        return any(
            build_seq > restage_seq and embedded >= staged_value
            for restage_seq, staged_value in staged
            for build_seq, embedded in built
        )

    return predicate


_WINDOW_SAFE_PREDICATES: Final[frozenset[str]] = frozenset(
    {
        "distinct_paths_at_least",
        "event_type_present",
        "matching_command_count",
        "writes_under_prefix",
    }
)


ACTION_REGISTRY: Final[dict[str, ActionPredicateFactory]] = {
    **{name: _windowed(REGISTRY[name]) for name in _WINDOW_SAFE_PREDICATES},
    "post_detection_search_verification": post_detection_search_verification,
    "restage_then_verified_build": restage_then_verified_build,
    "tool_call_count": tool_call_count,
}
