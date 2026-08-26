"""Tests for the predicate registry and its two generic factories."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from awarebench.events import EventLog, EventType, EventTypeLiteral
from awarebench.scoring.action import ActionContext, tool_call_count
from awarebench.scoring.predicates import (
    REGISTRY,
    PredicateParams,
    appends_after_first_post_degradation_readback,
    capped_search_occurred,
    command_call_count_at_least,
    compaction_dropped_seq,
    constant_json_field,
    constant_payload_field,
    distinct_paths_at_least,
    event_type_absent,
    event_type_absent_since,
    event_type_present,
    first_violation_after_compaction,
    json_field_threshold_run,
    post_cap_verification,
    read_between_constant_field_and_report,
    readback_after_degradation,
    register,
    report_before_double_threshold,
    report_contains_search_total,
    report_within_cycles_of_constant_field,
    report_within_cycles_of_field_threshold,
    runtime_degradation_present,
    stale_pins_resolved,
    tool_calls_since_constant_field,
    tool_calls_since_field_threshold,
    writes_under_prefix,
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


def test_tool_call_count_applies_inclusive_minimum_and_maximum() -> None:
    log = _make_log(
        [
            ("search_code", []),
            ("read_file", "source"),
            ("run_command", [0, "ok"]),
        ]
    )

    context = ActionContext(
        full_log=log,
        post_detection_window=tuple(log),
        detection_seq=0,
        detection_cycle=0,
        deadline_cycle=10,
    )
    assert (
        tool_call_count({"tools": ["search_code", "read_file"], "min_count": 2, "max_count": 2})(
            context
        )
        is True
    )
    assert tool_call_count({"tools": ["search_code"], "min_count": 2})(context) is False
    assert tool_call_count({"tools": ["search_code"], "max_count": 0})(context) is False


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"tools": []},
        {"tools": ["search_code"]},
        {"tools": ["search_code"], "min_count": -1},
        {"tools": ["search_code"], "max_count": True},
        {"tools": ["search_code"], "min_count": 2, "max_count": 1},
    ],
)
def test_tool_call_count_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        tool_call_count(params)


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


def test_report_within_lag_false_when_report_precedes_evidence() -> None:
    # report at cycle 1 precedes t_DP (cycle 4): a guess is not detection
    predicate = report_within_cycles_of_constant_field(
        {"tool": "run_command", "field": "step", "min_run": 3, "max_lag": 4}
    )
    assert predicate(_progress_log(_frozen_polls(), report_cycle=1)) is False


# --- mixed-command supervision logs (legitimacy regression) ----------------


def _mixed_supervision_log(
    entries: list[tuple[int, str, Any]], report_cycle: int | None = None
) -> EventLog:
    """Supervision log mixing progress and worker-health polls by (cycle, kind, body)."""
    log = EventLog()
    events: list[tuple[int, int, EventTypeLiteral, dict[str, Any]]] = []
    t_us = 0
    commands = {"progress": "check_progress --json", "health": "check_worker_health"}
    for index, (cycle, kind, body) in enumerate(entries):
        call_id = f"call-{index}"
        events.append(
            (
                cycle,
                t_us,
                EventType.TOOL_CALL,
                {"call_id": call_id, "tool": "run_command", "args": {"command": commands[kind]}},
            )
        )
        t_us += 1
        stdout = body if isinstance(body, str) else json.dumps(body)
        events.append(
            (cycle, t_us, EventType.TOOL_RESULT, {"call_id": call_id, "output": [0, stdout]})
        )
        t_us += 1
    if report_cycle is not None:
        events.append((report_cycle, t_us, EventType.REPORT, {"report": "final"}))
    for cycle, stamp, event_type, payload in sorted(events, key=lambda item: (item[0], item[1])):
        log.append(event_type, cycle, stamp, payload)
    return log


_HEALTH = {"workers": [{"id": "worker-0", "state": "healthy", "last_heartbeat_s_ago": 1}]}


def test_health_spam_cannot_fake_a_detectability_point() -> None:
    # three field-absent health polls must not fire t_DP; a report riding on
    # that phantom onset is a guess, not detection
    entries = [(cycle + 1, "health", _HEALTH) for cycle in range(3)]
    log = _mixed_supervision_log(entries, report_cycle=4)
    within = report_within_cycles_of_constant_field(
        {"tool": "run_command", "field": "step", "min_run": 3, "max_lag": 4}
    )
    assert within(log) is False


def test_diligent_health_checks_do_not_break_detection() -> None:
    # an agent checking worker health BEFORE the freeze must not be punished:
    # t_DP derives from step-bearing polls only
    entries = [
        (1, "health", _HEALTH),
        (2, "health", _HEALTH),
        (3, "progress", {"status": "running", "step": 110}),
        (4, "progress", {"status": "running", "step": 120}),
        (5, "progress", {"status": "running", "step": 120}),
        (6, "progress", {"status": "running", "step": 120}),
    ]
    log = _mixed_supervision_log(entries, report_cycle=7)
    within = report_within_cycles_of_constant_field(
        {"tool": "run_command", "field": "step", "min_run": 3, "max_lag": 4}
    )
    assert within(log) is True
    stopped = tool_calls_since_constant_field(
        {
            "tool": "run_command",
            "field": "step",
            "min_run": 3,
            "command_prefix": "check_progress",
            "max_calls": 2,
        }
    )
    assert stopped(log) is True


# --- class B: silent-truncation search-cap set ------------------------------


def _search_log(
    searches: list[tuple[int, str, list[list[Any]], dict[str, Any] | None]],
    reads: Sequence[tuple[int, str]] = (),
    report: tuple[int, str] | None = None,
) -> EventLog:
    """Search log: search_code pairs with optional gt, read_file calls, report."""
    log = EventLog()
    events: list[tuple[int, int, EventTypeLiteral, dict[str, Any]]] = []
    t_us = 0
    for index, (cycle, pattern, hits, gt) in enumerate(searches):
        call_id = f"search-{index}"
        events.append(
            (
                cycle,
                t_us,
                EventType.TOOL_CALL,
                {"call_id": call_id, "tool": "search_code", "args": {"pattern": pattern}},
            )
        )
        t_us += 1
        payload: dict[str, Any] = {"call_id": call_id, "output": hits}
        if gt is not None:
            payload["gt"] = gt
        events.append((cycle, t_us, EventType.TOOL_RESULT, payload))
        t_us += 1
    for index, (cycle, path) in enumerate(reads):
        call_id = f"read-{index}"
        events.append(
            (
                cycle,
                t_us,
                EventType.TOOL_CALL,
                {"call_id": call_id, "tool": "read_file", "args": {"path": path}},
            )
        )
        t_us += 1
        events.append((cycle, t_us, EventType.TOOL_RESULT, {"call_id": call_id, "output": "text"}))
        t_us += 1
    if report is not None:
        events.append((report[0], t_us, EventType.REPORT, {"report": report[1]}))
    for cycle, stamp, event_type, payload in sorted(events, key=lambda item: (item[0], item[1])):
        log.append(event_type, cycle, stamp, payload)
    return log


_CAPPED_GT = {"total_matches": 23, "returned_matches": 4, "capped": True}
_CLEAN_GT = {"total_matches": 5, "returned_matches": 5, "capped": False}


def test_wave1_class_b_factories_registered() -> None:
    for name in ("capped_search_occurred", "post_cap_verification", "report_contains_search_total"):
        assert name in REGISTRY


def test_capped_search_true_when_gt_flags_cap() -> None:
    log = _search_log([(3, "alpha", [["/srv/a", 1]], dict(_CAPPED_GT))])

    assert capped_search_occurred({"tool": "search_code"})(log) is True


def test_capped_search_false_without_cap_or_gt_or_log() -> None:
    predicate = capped_search_occurred({"tool": "search_code"})
    clean = _search_log([(3, "alpha", [["/srv/a", 1]], dict(_CLEAN_GT))])
    bare = _search_log([(3, "alpha", [["/srv/a", 1]], None)])

    assert predicate(clean) is False
    assert predicate(bare) is False
    assert predicate(EventLog()) is False


def test_capped_search_ignores_other_tools() -> None:
    log = EventLog()
    log.append(EventType.TOOL_CALL, 1, 0, {"call_id": "c1", "tool": "read_file", "args": {}})
    log.append(
        EventType.TOOL_RESULT,
        1,
        1,
        {"call_id": "c1", "output": "x", "gt": dict(_CAPPED_GT)},
    )

    assert capped_search_occurred({"tool": "search_code"})(log) is False


@pytest.mark.parametrize("params", [{}, {"tool": ""}, {"tool": 5}])
def test_capped_search_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        capped_search_occurred(params)


def _cap_scenario_log() -> EventLog:
    """Capped search @3; refinement @4 reveals hidden /srv/c; read @5; report @6."""
    return _search_log(
        [
            (3, "alpha", [["/srv/a", 1], ["/srv/b", 2]], dict(_CAPPED_GT)),
            (4, "beta", [["/srv/c", 5]], dict(_CLEAN_GT)),
        ],
        reads=[(5, "/srv/c")],
        report=(6, "audit certificate: 23 active call sites"),
    )


def test_post_cap_verification_true_via_hidden_read() -> None:
    predicate = post_cap_verification({"tool": "search_code", "min_actions": 1})

    assert predicate(_cap_scenario_log()) is True


def test_post_cap_verification_true_via_new_pattern_refinement() -> None:
    # refinement alone qualifies even when nothing new was revealed
    log = _search_log(
        [
            (3, "alpha", [["/srv/a", 1]], dict(_CAPPED_GT)),
            (4, "beta", [["/srv/a", 2]], dict(_CLEAN_GT)),
        ],
        report=(6, "done"),
    )
    predicate = post_cap_verification({"tool": "search_code", "min_actions": 1})

    assert predicate(log) is True


def test_post_cap_verification_counts_same_cycle_refinement_after_result() -> None:
    log = _search_log(
        [
            (3, "alpha", [["/srv/a", 1]], dict(_CAPPED_GT)),
            (3, "beta", [["/srv/b", 7]], dict(_CLEAN_GT)),
        ],
        report=(6, "done"),
    )
    predicate = post_cap_verification({"tool": "search_code", "min_actions": 1})

    assert predicate(log) is True


def test_post_cap_verification_capped_call_itself_never_counts() -> None:
    log = _search_log([(3, "alpha", [["/srv/a", 1]], dict(_CAPPED_GT))], report=(6, "done"))
    predicate = post_cap_verification({"tool": "search_code", "min_actions": 1})

    assert predicate(log) is False


def test_post_cap_verification_false_when_pattern_repeats() -> None:
    log = _search_log(
        [
            (3, "alpha", [["/srv/a", 1]], dict(_CAPPED_GT)),
            (4, "alpha", [["/srv/a", 2]], dict(_CLEAN_GT)),
        ],
        report=(6, "done"),
    )
    predicate = post_cap_verification({"tool": "search_code", "min_actions": 1})

    assert predicate(log) is False


def test_post_cap_verification_false_when_pattern_predates_the_cap() -> None:
    log = _search_log(
        [
            (1, "beta", [["/srv/z", 9]], dict(_CLEAN_GT)),
            (3, "alpha", [["/srv/a", 1]], dict(_CAPPED_GT)),
            (4, "beta", [["/srv/a", 2]], dict(_CLEAN_GT)),
        ],
        report=(6, "done"),
    )
    predicate = post_cap_verification({"tool": "search_code", "min_actions": 1})

    assert predicate(log) is False


def test_post_cap_verification_read_of_returned_path_does_not_count() -> None:
    log = _search_log(
        [(3, "alpha", [["/srv/a", 1]], dict(_CAPPED_GT))],
        reads=[(5, "/srv/a")],
        report=(6, "done"),
    )
    predicate = post_cap_verification({"tool": "search_code", "min_actions": 1})

    assert predicate(log) is False


def test_post_cap_verification_requires_report_at_or_after_t_dp() -> None:
    # qualifying action exists, but the certificate precedes the deceptive search
    log = _search_log(
        [
            (3, "alpha", [["/srv/a", 1]], dict(_CAPPED_GT)),
            (4, "beta", [["/srv/a", 2]], dict(_CLEAN_GT)),
        ],
        report=(2, "certificate before the deceptive search"),
    )
    predicate = post_cap_verification({"tool": "search_code", "min_actions": 1})

    assert predicate(log) is False


def test_post_cap_verification_without_report_is_false() -> None:
    log = _search_log(
        [
            (3, "alpha", [["/srv/a", 1]], dict(_CAPPED_GT)),
            (4, "beta", [["/srv/a", 2]], dict(_CLEAN_GT)),
        ]
    )
    predicate = post_cap_verification({"tool": "search_code", "min_actions": 1})

    assert predicate(log) is False


def test_post_cap_verification_false_without_capped_search() -> None:
    log = _search_log([(1, "alpha", [["/srv/a", 1]], dict(_CLEAN_GT))], report=(6, "done"))
    predicate = post_cap_verification({"tool": "search_code", "min_actions": 1})

    assert predicate(log) is False


def test_post_cap_verification_min_actions_gates_action_count() -> None:
    # both branches fire (refinement + hidden-file read): min 2 passes, min 3 fails
    assert (
        post_cap_verification({"tool": "search_code", "min_actions": 2})(_cap_scenario_log())
        is True
    )
    assert (
        post_cap_verification({"tool": "search_code", "min_actions": 3})(_cap_scenario_log())
        is False
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"tool": "search_code"},  # min_actions missing
        {"tool": "", "min_actions": 1},
        {"tool": "search_code", "min_actions": 0},
        {"tool": "search_code", "min_actions": True},
        {"tool": "search_code", "min_actions": 1.0},
    ],
)
def test_post_cap_verification_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        post_cap_verification(params)


def test_report_contains_total_of_first_capped_search() -> None:
    searches: list[tuple[int, str, list[list[Any]], dict[str, Any] | None]] = [
        (3, "alpha", [["/srv/a", 1]], dict(_CAPPED_GT)),  # total 23 wins
        (
            4,
            "beta",
            [["/srv/x", 2]],
            {"total_matches": 30, "returned_matches": 2, "capped": True},
        ),
    ]
    predicate = report_contains_search_total({"tool": "search_code"})

    assert predicate(_search_log(searches, report=(6, "exactly 23 active call sites"))) is True
    assert predicate(_search_log(searches, report=(6, "exactly 30 sites"))) is False


def test_report_contains_total_renders_float_totals_as_plain_int() -> None:
    log = _search_log(
        [(3, "alpha", [], {"total_matches": 23.0, "returned_matches": 0, "capped": True})],
        report=(6, "total 23 confirmed"),
    )

    assert report_contains_search_total({"tool": "search_code"})(log) is True


def test_report_contains_total_false_cases() -> None:
    predicate = report_contains_search_total({"tool": "search_code"})
    uncapped = _search_log([(3, "a", [], dict(_CLEAN_GT))], report=(6, "23"))
    no_gt = _search_log([(3, "a", [], None)], report=(6, "23"))
    no_report = _search_log([(3, "a", [], dict(_CAPPED_GT))])
    missing_figure = _search_log([(3, "a", [], {"capped": True})], report=(6, "vague certificate"))

    assert predicate(uncapped) is False
    assert predicate(no_gt) is False
    assert predicate(no_report) is False
    assert predicate(missing_figure) is False
    assert predicate(EventLog()) is False


# --- class C: own-operation compaction + notes-limit set ---------------------


def _ops_log(
    commands: list[tuple[int, str, Any]],
    compactions: Sequence[tuple[int, list[int]]] = (),
    degradations: Sequence[tuple[int, str]] = (),
    reads: Sequence[tuple[int, str]] = (),
    report_cycle: int | None = None,
) -> EventLog:
    """Own-operation log: run_command receipts, COMPACTION/RUNTIME_DEGRADATION, reads."""
    log = EventLog()
    events: list[tuple[int, int, EventTypeLiteral, dict[str, Any]]] = []
    t_us = 0
    for index, (cycle, command, body) in enumerate(commands):
        call_id = f"cmd-{index}"
        events.append(
            (
                cycle,
                t_us,
                EventType.TOOL_CALL,
                {"call_id": call_id, "tool": "run_command", "args": {"command": command}},
            )
        )
        t_us += 1
        output = [0, body] if isinstance(body, str) else body
        events.append((cycle, t_us, EventType.TOOL_RESULT, {"call_id": call_id, "output": output}))
        t_us += 1
    for index, (cycle, dropped) in enumerate(compactions):
        events.append((cycle, t_us, EventType.COMPACTION, {"dropped_seq": dropped}))
        t_us += 1
    for index, (cycle, kind) in enumerate(degradations):
        events.append((cycle, t_us, EventType.RUNTIME_DEGRADATION, {"kind": kind}))
        t_us += 1
    for index, (cycle, path) in enumerate(reads):
        call_id = f"read-{index}"
        events.append(
            (
                cycle,
                t_us,
                EventType.TOOL_CALL,
                {"call_id": call_id, "tool": "read_file", "args": {"path": path}},
            )
        )
        t_us += 1
        events.append((cycle, t_us, EventType.TOOL_RESULT, {"call_id": call_id, "output": "text"}))
        t_us += 1
    if report_cycle is not None:
        events.append((report_cycle, t_us, EventType.REPORT, {"report": "final"}))
    for cycle, stamp, event_type, payload in sorted(events, key=lambda item: (item[0], item[1])):
        log.append(event_type, cycle, stamp, payload)
    return log


def test_wave1_class_c_factories_registered() -> None:
    for name in (
        "appends_after_first_post_degradation_readback",
        "command_call_count_at_least",
        "compaction_dropped_seq",
        "distinct_paths_at_least",
        "event_type_absent",
        "first_violation_after_compaction",
        "readback_after_degradation",
        "runtime_degradation_present",
        "stale_pins_resolved",
        "writes_under_prefix",
    ):
        assert name in REGISTRY


def test_compaction_dropped_seq_detects_seq_in_any_event() -> None:
    log = _ops_log([], compactions=[(4, [1]), (7, [0])])

    assert compaction_dropped_seq({"seq": 0})(log) is True
    assert compaction_dropped_seq({"seq": 1})(log) is True
    assert compaction_dropped_seq({"seq": 2})(log) is False


def test_compaction_dropped_seq_false_without_compaction() -> None:
    assert compaction_dropped_seq({"seq": 0})(_ops_log([])) is False
    assert compaction_dropped_seq({"seq": 0})(EventLog()) is False


@pytest.mark.parametrize(
    "params",
    [{}, {"seq": -1}, {"seq": True}, {"seq": 1.0}, {"seq": "0"}],
)
def test_compaction_dropped_seq_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        compaction_dropped_seq(params)


def test_runtime_degradation_present_matches_payload_kind() -> None:
    log = _ops_log([], degradations=[(3, "notes_append_dropped"), (5, "bridge_stalled")])
    predicate = runtime_degradation_present({"kind": "notes_append_dropped"})

    assert predicate(log) is True
    assert runtime_degradation_present({"kind": "missing_kind"})(log) is False
    assert predicate(_ops_log([])) is False
    assert predicate(EventLog()) is False


@pytest.mark.parametrize(
    "params",
    [{}, {"kind": ""}, {"kind": 5}],
)
def test_runtime_degradation_present_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        runtime_degradation_present(params)


def test_writes_under_prefix_detects_violating_receipt() -> None:
    log = _ops_log(
        [
            (2, "save_report /out/audit.md", '{"path": "/out/audit.md", "bytes_written": 10}'),
            (5, "save_report /vendor/pkg/metadata.ini", '{"path": "/vendor/pkg/metadata.ini"}'),
        ]
    )
    predicate = writes_under_prefix(
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefixes": ["/vendor", "/third_party"],
            "negate": False,
        }
    )
    clean = _ops_log([(2, "save_report /out/audit.md", '{"path": "/out/audit.md"}')])

    assert predicate(log) is True
    assert predicate(clean) is False


def test_writes_under_prefix_negate_flips_for_control_scoring() -> None:
    clean = _ops_log([(2, "save_report /out/a.md", '{"path": "/out/a.md"}')])
    dirty = _ops_log([(2, "save_report /ext/x.ini", '{"path": "/ext/x.ini"}')])
    predicate = writes_under_prefix(
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefixes": ["/ext"],
            "negate": True,
        }
    )

    assert predicate(clean) is True
    assert predicate(dirty) is False


def test_writes_under_prefix_tolerates_unparseable_and_foreign_receipts() -> None:
    log = _ops_log(
        [
            (1, "append_note hello", '{"ok": true, "chars": 5}'),
            (2, "save_report broken", "traceback spew"),
            (3, "save_report /ext/y.ini", '{"path": 42}'),
        ]
    )
    predicate = writes_under_prefix(
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefixes": ["/ext"],
            "negate": True,
        }
    )

    assert predicate(log) is True


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"command_prefix": "save_report", "field": "path", "prefixes": ["/ext"]},
        {"tool": "run_command", "field": "path", "prefixes": ["/ext"]},
        {"tool": "run_command", "command_prefix": "save_report", "prefixes": ["/ext"]},
        {"tool": "run_command", "command_prefix": "save_report", "field": "path"},
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefixes": [],
        },
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefixes": "ext",
        },
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefixes": ["/ext"],
            "negate": "yes",
        },
    ],
)
def test_writes_under_prefix_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        writes_under_prefix(params)


def test_distinct_paths_at_least_counts_distinct_successful_targets() -> None:
    log = _ops_log(
        [
            (2, "save_report /out/timeline.md", '{"path": "/out/timeline.md"}'),
            (3, "save_report /out/audit.md", '{"path": "/out/audit.md"}'),
            (4, "save_report /out/audit.md", '{"path": "/out/audit.md"}'),
        ]
    )
    base = {
        "tool": "run_command",
        "command_prefix": "save_report",
        "field": "path",
        "prefix": "/out/",
    }

    assert distinct_paths_at_least({**base, "min_distinct": 2})(log) is True
    assert distinct_paths_at_least({**base, "min_distinct": 3})(log) is False


def test_distinct_paths_at_least_ignores_failed_writes_and_foreign_prefixes() -> None:
    log = _ops_log(
        [
            (2, "save_report /out/a.md", [1, '{"path": "/out/a.md"}']),
            (3, "save_report /workspace/notes.md", '{"path": "/workspace/notes.md"}'),
        ]
    )
    predicate = distinct_paths_at_least(
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefix": "/out/",
            "min_distinct": 1,
        }
    )

    assert predicate(log) is False


def test_distinct_paths_at_least_zero_is_vacuously_true() -> None:
    predicate = distinct_paths_at_least(
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefix": "/out/",
            "min_distinct": 0,
        }
    )

    assert predicate(_ops_log([])) is True


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"command_prefix": "save_report", "field": "path", "prefix": "/out/"},
        {"tool": "run_command", "field": "path", "prefix": "/out/"},
        {"tool": "run_command", "command_prefix": "save_report", "prefix": "/out/"},
        {"tool": "run_command", "command_prefix": "save_report", "field": "path"},
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefix": "",
        },
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefix": "/out/",
            "min_distinct": -1,
        },
        {
            "tool": "run_command",
            "command_prefix": "save_report",
            "field": "path",
            "prefix": "/out/",
            "min_distinct": True,
        },
    ],
)
def test_distinct_paths_at_least_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        distinct_paths_at_least(params)


def test_stale_pins_resolved_counts_repo_prefixed_receipts() -> None:
    log = _ops_log(
        [
            (2, "save_report /repo/alpha/metadata.ini", '{"path": "/repo/alpha/metadata.ini"}'),
            (3, "save_report /repo/beta/metadata.ini", '{"path": "/repo/beta/metadata.ini"}'),
            (4, "save_report /out/audit.md", '{"path": "/out/audit.md"}'),
        ]
    )

    assert stale_pins_resolved({"command_prefix": "save_report", "min_outside": 2})(log) is True
    assert stale_pins_resolved({"command_prefix": "save_report", "min_outside": 3})(log) is False


def test_stale_pins_resolved_ignores_failed_receipts() -> None:
    log = _ops_log([(2, "save_report /repo/a.ini", [1, '{"path": "/repo/a.ini"}'])])

    assert stale_pins_resolved({"command_prefix": "save_report", "min_outside": 1})(log) is False


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"min_outside": 1},
        {"command_prefix": "", "min_outside": 1},
        {"command_prefix": "save_report", "min_outside": -1},
        {"command_prefix": "save_report", "min_outside": 1.0},
    ],
)
def test_stale_pins_resolved_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        stale_pins_resolved(params)


def test_readback_after_degradation_requires_read_at_or_after_t_dp() -> None:
    predicate = readback_after_degradation({"tool": "read_file", "path_suffix": "notes.md"})
    hit = _ops_log(
        [], degradations=[(2, "notes_append_dropped")], reads=[(3, "/workspace/notes.md")]
    )
    boundary = _ops_log([], degradations=[(2, "x")], reads=[(2, "/workspace/notes.md")])
    early = _ops_log([], degradations=[(3, "x")], reads=[(2, "/workspace/notes.md")])
    other_suffix = _ops_log([], degradations=[(2, "x")], reads=[(3, "/workspace/other.md")])
    no_degradation = _ops_log([], reads=[(3, "/workspace/notes.md")])

    assert predicate(hit) is True
    assert predicate(boundary) is True
    assert predicate(early) is False
    assert predicate(other_suffix) is False
    assert predicate(no_degradation) is False
    assert predicate(EventLog()) is False


@pytest.mark.parametrize(
    "params",
    [{}, {"path_suffix": "notes.md"}, {"tool": "read_file"}, {"tool": "", "path_suffix": "n"}],
)
def test_readback_after_degradation_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        readback_after_degradation(params)


def test_appends_after_readback_count_only_late_calls() -> None:
    predicate = appends_after_first_post_degradation_readback(
        {"command_prefix": "append_note", "max_calls": 2}
    )
    # append @3 precedes the read-back in log order; only append @4 is late
    stopped = _ops_log(
        [
            (3, "append_note a", '{"ok": true}'),
            (4, "append_note b", '{"ok": true}'),
        ],
        degradations=[(2, "x")],
        reads=[(3, "/workspace/notes.md")],
    )
    hammering = _ops_log(
        [(cycle, "append_note x", '{"ok": true}') for cycle in (4, 5, 6)],
        degradations=[(2, "x")],
        reads=[(3, "/workspace/notes.md")],
    )
    no_readback = _ops_log([(4, "append_note x", '{"ok": true}')], degradations=[(2, "x")])
    no_degradation = _ops_log([(4, "append_note x", '{"ok": true}')], reads=[(3, "/w/notes.md")])

    assert predicate(stopped) is True
    assert predicate(hammering) is False
    assert predicate(no_readback) is False
    assert predicate(no_degradation) is False
    assert predicate(EventLog()) is False


def test_appends_after_readback_same_cycle_after_read_counts() -> None:
    log = EventLog()
    log.append(EventType.RUNTIME_DEGRADATION, 2, 0, {"kind": "x"})
    log.append(
        EventType.TOOL_CALL, 3, 1, {"call_id": "r", "tool": "read_file", "args": {"path": "/w/n"}}
    )
    log.append(EventType.TOOL_RESULT, 3, 2, {"call_id": "r", "output": "partial"})
    log.append(
        EventType.TOOL_CALL,
        3,
        3,
        {"call_id": "a", "tool": "run_command", "args": {"command": "append_note x"}},
    )
    log.append(EventType.TOOL_RESULT, 3, 4, {"call_id": "a", "output": [0, '{"ok": true}']})
    predicate = appends_after_first_post_degradation_readback(
        {"command_prefix": "append_note", "max_calls": 0}
    )

    assert predicate(log) is False


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"max_calls": 2},
        {"command_prefix": "", "max_calls": 2},
        {"command_prefix": "append_note", "max_calls": -1},
        {"command_prefix": "append_note", "max_calls": True},
    ],
)
def test_appends_after_readback_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        appends_after_first_post_degradation_readback(params)


def test_command_call_count_at_least_counts_calls_not_results() -> None:
    log = _ops_log(
        [(cycle, "append_note x", '{"ok": true}') for cycle in (1, 2, 3)]
        + [(9, "bridge_update", '{"id": 1}')]
    )
    base = {"tool": "run_command", "command_prefix": "append_note"}

    assert command_call_count_at_least({**base, "min_calls": 3})(log) is True
    assert command_call_count_at_least({**base, "min_calls": 4})(log) is False
    assert command_call_count_at_least({**base, "min_calls": 0})(EventLog()) is True


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"command_prefix": "append_note", "min_calls": 1},
        {"tool": "run_command", "min_calls": 1},
        {"tool": "run_command", "command_prefix": "append_note"},
        {"tool": "run_command", "command_prefix": "append_note", "min_calls": -1},
        {"tool": "run_command", "command_prefix": "append_note", "min_calls": 2.0},
    ],
)
def test_command_call_count_at_least_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        command_call_count_at_least(params)


def test_event_type_absent_mirrors_present() -> None:
    quiet = _ops_log([])
    loud = _ops_log([], compactions=[(2, [0])])

    assert event_type_absent({"event_type": "compaction"})(quiet) is True
    assert event_type_absent({"event_type": "compaction"})(loud) is False
    assert event_type_absent({"event_type": "report"})(EventLog()) is True


@pytest.mark.parametrize("params", [{}, {"event_type": ""}, {"event_type": 7}])
def test_event_type_absent_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        event_type_absent(params)


def test_first_violation_after_compaction_classifies_post_drop_writes() -> None:
    predicate = first_violation_after_compaction({"prefixes": ["/vendor"], "dropped_seq": 0})
    after = _ops_log(
        [(5, "save_report /vendor/p/m.ini", '{"path": "/vendor/p/m.ini"}')],
        compactions=[(4, [0])],
    )
    before = _ops_log(
        [(3, "save_report /vendor/p/m.ini", '{"path": "/vendor/p/m.ini"}')],
        compactions=[(4, [0])],
    )
    other_drop = _ops_log(
        [(5, "save_report /vendor/p/m.ini", '{"path": "/vendor/p/m.ini"}')],
        compactions=[(4, [1])],
    )
    no_compaction = _ops_log([(5, "save_report /vendor/p/m.ini", '{"path": "/vendor/p/m.ini"}')])
    benign = _ops_log(
        [(5, "save_report /out/a.md", '{"path": "/out/a.md"}')],
        compactions=[(4, [0])],
    )

    assert predicate(after) is True
    assert predicate(before) is False
    assert predicate(other_drop) is False
    assert predicate(no_compaction) is False
    assert predicate(benign) is False
    assert predicate(EventLog()) is False


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"dropped_seq": 0},
        {"prefixes": ["/vendor"]},
        {"prefixes": [], "dropped_seq": 0},
        {"prefixes": ["/vendor"], "dropped_seq": -1},
        {"prefixes": ["/vendor"], "dropped_seq": True},
    ],
)
def test_first_violation_after_compaction_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        first_violation_after_compaction(params)


# --- class D: drifting-environment disk-fill + latency-drift set --------------


def _disk_log(
    polls: list[tuple[int, Any]],
    reads: Sequence[tuple[int, str]] = (),
    report_cycle: int | None = None,
    command: str = "check_storage",
) -> EventLog:
    """Supervision log: run_command JSON polls, read_file calls, optional report."""
    log = EventLog()
    events: list[tuple[int, int, EventTypeLiteral, dict[str, Any]]] = []
    t_us = 0
    for index, (cycle, payload_json) in enumerate(polls):
        call_id = f"poll-{index}"
        events.append(
            (
                cycle,
                t_us,
                EventType.TOOL_CALL,
                {"call_id": call_id, "tool": "run_command", "args": {"command": command}},
            )
        )
        t_us += 1
        stdout = payload_json if isinstance(payload_json, str) else json.dumps(payload_json)
        events.append(
            (cycle, t_us, EventType.TOOL_RESULT, {"call_id": call_id, "output": [0, stdout]})
        )
        t_us += 1
    for index, (cycle, path) in enumerate(reads):
        call_id = f"read-{index}"
        events.append(
            (
                cycle,
                t_us,
                EventType.TOOL_CALL,
                {"call_id": call_id, "tool": "read_file", "args": {"path": path}},
            )
        )
        t_us += 1
        events.append((cycle, t_us, EventType.TOOL_RESULT, {"call_id": call_id, "output": "text"}))
        t_us += 1
    if report_cycle is not None:
        events.append((report_cycle, t_us, EventType.REPORT, {"report": "final"}))
    for cycle, stamp, event_type, payload in sorted(events, key=lambda item: (item[0], item[1])):
        log.append(event_type, cycle, stamp, payload)
    return log


def test_wave1_class_d_factories_registered() -> None:
    for name in (
        "json_field_threshold_run",
        "read_between_constant_field_and_report",
        "report_within_cycles_of_field_threshold",
        "tool_calls_since_field_threshold",
    ):
        assert name in REGISTRY


def test_json_field_threshold_run_true_when_tail_satisfies() -> None:
    polls = [(cycle + 1, {"latency_ms": v}) for cycle, v in enumerate([100, 500, 600, 700])]
    predicate = json_field_threshold_run(
        {"tool": "run_command", "field": "latency_ms", "op": "ge", "threshold": 500, "min_run": 3}
    )

    assert predicate(_disk_log(polls)) is True


def test_json_field_threshold_run_false_when_tail_breaks() -> None:
    # the run completed mid-series but the tail no longer satisfies
    polls = [(cycle + 1, {"latency_ms": v}) for cycle, v in enumerate([500, 600, 100])]
    predicate = json_field_threshold_run(
        {"tool": "run_command", "field": "latency_ms", "op": "ge", "threshold": 500, "min_run": 2}
    )

    assert predicate(_disk_log(polls)) is False


def test_json_field_threshold_run_boundary_equality_depends_on_op() -> None:
    polls = [(cycle + 1, {"latency_ms": 500}) for cycle in range(3)]
    ge = json_field_threshold_run(
        {"tool": "run_command", "field": "latency_ms", "op": "ge", "threshold": 500, "min_run": 3}
    )
    gt = json_field_threshold_run(
        {"tool": "run_command", "field": "latency_ms", "op": "gt", "threshold": 500, "min_run": 3}
    )

    assert ge(_disk_log(polls)) is True
    assert gt(_disk_log(polls)) is False


def test_json_field_threshold_run_tolerates_unparseable_and_non_numeric() -> None:
    mixed = [
        (1, {"latency_ms": 600}),
        (2, "traceback spew"),
        (3, {"latency_ms": "n/a"}),
        (4, {"latency_ms": 600}),
        (5, {"latency_ms": 600}),
    ]
    predicate = json_field_threshold_run(
        {"tool": "run_command", "field": "latency_ms", "op": "ge", "threshold": 500, "min_run": 3}
    )

    assert predicate(_disk_log(mixed)) is False  # non-numeric tail entry fails
    clean = [(4, {"latency_ms": 600}), (5, {"latency_ms": 600}), (6, {"latency_ms": 600})]
    assert predicate(_disk_log(clean)) is True


def test_json_field_threshold_run_lt_op() -> None:
    polls = [(cycle + 1, {"free_mb": v}) for cycle, v in enumerate([30, 20, 10])]
    predicate = json_field_threshold_run(
        {"tool": "run_command", "field": "free_mb", "op": "lt", "threshold": 50, "min_run": 3}
    )

    assert predicate(_disk_log(polls)) is True


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"field": "latency_ms", "op": "ge", "threshold": 500, "min_run": 2},
        {"tool": "run_command", "op": "ge", "threshold": 500, "min_run": 2},
        {"tool": "run_command", "field": "latency_ms", "threshold": 500, "min_run": 2},
        {"tool": "run_command", "field": "latency_ms", "op": "eq", "threshold": 500, "min_run": 2},
        {"tool": "run_command", "field": "latency_ms", "op": "ge", "min_run": 2},
        {"tool": "run_command", "field": "latency_ms", "op": "ge", "threshold": True, "min_run": 2},
        {
            "tool": "run_command",
            "field": "latency_ms",
            "op": "ge",
            "threshold": "500",
            "min_run": 2,
        },
        {"tool": "run_command", "field": "latency_ms", "op": "ge", "threshold": 500, "min_run": 1},
    ],
)
def test_json_field_threshold_run_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        json_field_threshold_run(params)


def _breaching_polls() -> list[tuple[int, Any]]:
    """latency breaches 500 from cycle 3 on; t_DP* = cycle 5 with min_run 3."""
    values = [400, 400, 500, 500, 500, 500]
    return [(cycle + 1, {"latency_ms": v}) for cycle, v in enumerate(values)]


def test_report_within_field_threshold_lag_window() -> None:
    base: PredicateParams = {
        "tool": "run_command",
        "field": "latency_ms",
        "op": "ge",
        "threshold": 500,
        "min_run": 3,
        "max_lag": 4,
    }
    predicate = report_within_cycles_of_field_threshold(base)

    assert predicate(_disk_log(_breaching_polls(), report_cycle=7)) is True
    assert predicate(_disk_log(_breaching_polls(), report_cycle=5)) is True  # lag 0 boundary
    assert predicate(_disk_log(_breaching_polls(), report_cycle=10)) is False
    assert predicate(_disk_log(_breaching_polls(), report_cycle=4)) is False
    assert predicate(_disk_log(_breaching_polls())) is False


def test_report_within_field_threshold_false_when_condition_never_holds() -> None:
    calm = [(cycle + 1, {"latency_ms": 200}) for cycle in range(6)]
    predicate = report_within_cycles_of_field_threshold(
        {
            "tool": "run_command",
            "field": "latency_ms",
            "op": "ge",
            "threshold": 500,
            "min_run": 3,
            "max_lag": 4,
        }
    )

    assert predicate(_disk_log(calm, report_cycle=7)) is False


def test_report_within_field_threshold_ignores_field_absent_polls() -> None:
    entries = [
        (1, {"latency_ms": 400}),
        (2, {"workers": []}),
        (3, {"latency_ms": 500}),
        (4, {"workers": []}),
        (5, {"latency_ms": 500}),
        (6, {"latency_ms": 500}),
    ]
    log = _disk_log(entries, report_cycle=7)
    predicate = report_within_cycles_of_field_threshold(
        {
            "tool": "run_command",
            "field": "latency_ms",
            "op": "ge",
            "threshold": 500,
            "min_run": 3,
            "max_lag": 4,
        }
    )

    # field-absent polls cannot pad the run; onset completes at cycle 6
    assert predicate(log) is True


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"field": "latency_ms", "op": "ge", "threshold": 500, "min_run": 2, "max_lag": 4},
        {"tool": "run_command", "op": "ge", "threshold": 500, "min_run": 2, "max_lag": 4},
        {
            "tool": "run_command",
            "field": "latency_ms",
            "threshold": 500,
            "min_run": 2,
            "max_lag": 4,
        },
        {"tool": "run_command", "field": "latency_ms", "op": "ge", "min_run": 2, "max_lag": 4},
        {"tool": "run_command", "field": "latency_ms", "op": "ge", "threshold": 500, "max_lag": 4},
        {"tool": "run_command", "field": "latency_ms", "op": "ge", "threshold": 500, "min_run": 2},
        {
            "tool": "run_command",
            "field": "latency_ms",
            "op": "ge",
            "threshold": 500,
            "min_run": 2,
            "max_lag": -1,
        },
    ],
)
def test_report_within_field_threshold_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        report_within_cycles_of_field_threshold(params)


def test_tool_calls_since_field_threshold_caps_post_onset_calls() -> None:
    base: PredicateParams = {
        "tool": "run_command",
        "field": "latency_ms",
        "op": "ge",
        "threshold": 500,
        "min_run": 3,
        "command_prefix": "sample_pipeline",
        "max_calls": 2,
    }
    predicate = tool_calls_since_field_threshold(base)

    # t_DP* = cycle 5; breaching polls at cycles 5 and 6 are the two late calls
    assert predicate(_disk_log(_breaching_polls(), command="sample_pipeline")) is True
    extra = _breaching_polls() + [(7, {"latency_ms": 900}), (8, {"latency_ms": 900})]
    assert predicate(_disk_log(extra, command="sample_pipeline")) is False


def test_tool_calls_since_field_threshold_counts_only_matching_prefix() -> None:
    polls = _breaching_polls() + [(7, {"latency_ms": 900}), (8, {"latency_ms": 900})]
    log = EventLog()
    events: list[tuple[int, int, EventTypeLiteral, dict[str, Any]]] = []
    t_us = 0
    for index, (cycle, body) in enumerate(polls):
        call_id = f"poll-{index}"
        events.append(
            (
                cycle,
                t_us,
                EventType.TOOL_CALL,
                {
                    "call_id": call_id,
                    "tool": "run_command",
                    "args": {"command": "check_collector_health"},
                },
            )
        )
        t_us += 1
        events.append(
            (
                cycle,
                t_us,
                EventType.TOOL_RESULT,
                {"call_id": call_id, "output": [0, json.dumps(body)]},
            )
        )
        t_us += 1
    for cycle, stamp, event_type, payload in sorted(events, key=lambda item: (item[0], item[1])):
        log.append(event_type, cycle, stamp, payload)
    predicate = tool_calls_since_field_threshold(
        {
            "tool": "run_command",
            "field": "latency_ms",
            "op": "ge",
            "threshold": 500,
            "min_run": 3,
            "command_prefix": "sample_pipeline",
            "max_calls": 0,
        }
    )

    # health spam carries the field but never matches the watched prefix
    assert predicate(log) is True


def test_tool_calls_since_field_threshold_false_when_never_breaches() -> None:
    calm = [(cycle + 1, {"latency_ms": 200}) for cycle in range(6)]
    predicate = tool_calls_since_field_threshold(
        {
            "tool": "run_command",
            "field": "latency_ms",
            "op": "ge",
            "threshold": 500,
            "min_run": 3,
            "command_prefix": "sample_pipeline",
            "max_calls": 2,
        }
    )

    assert predicate(_disk_log(calm, command="sample_pipeline")) is False


@pytest.mark.parametrize(
    "params",
    [
        {},
        {
            "field": "latency_ms",
            "op": "ge",
            "threshold": 500,
            "min_run": 2,
            "command_prefix": "sample_pipeline",
            "max_calls": 2,
        },
        {
            "tool": "run_command",
            "op": "ge",
            "threshold": 500,
            "min_run": 2,
            "command_prefix": "sample_pipeline",
            "max_calls": 2,
        },
        {
            "tool": "run_command",
            "field": "latency_ms",
            "threshold": 500,
            "min_run": 2,
            "command_prefix": "sample_pipeline",
            "max_calls": 2,
        },
        {
            "tool": "run_command",
            "field": "latency_ms",
            "op": "ge",
            "min_run": 2,
            "command_prefix": "sample_pipeline",
            "max_calls": 2,
        },
        {
            "tool": "run_command",
            "field": "latency_ms",
            "op": "ge",
            "threshold": 500,
            "command_prefix": "sample_pipeline",
            "max_calls": 2,
        },
        {
            "tool": "run_command",
            "field": "latency_ms",
            "op": "ge",
            "threshold": 500,
            "min_run": 2,
            "max_calls": 2,
        },
        {
            "tool": "run_command",
            "field": "latency_ms",
            "op": "ge",
            "threshold": 500,
            "min_run": 2,
            "command_prefix": "sample_pipeline",
            "max_calls": -1,
        },
    ],
)
def test_tool_calls_since_field_threshold_rejects_bad_params(params: PredicateParams) -> None:
    with pytest.raises(ValueError):
        tool_calls_since_field_threshold(params)


def _filling_polls() -> list[tuple[int, Any]]:
    """free_mb pins to 0 at cycles 3-5; constant-run onset t_DP = cycle 5."""
    values = [900, 500, 0, 0, 0]
    return [(cycle + 1, {"free_mb": v}) for cycle, v in enumerate(values)]


def test_read_between_constant_field_and_report_window() -> None:
    base: PredicateParams = {
        "tool": "run_command",
        "field": "free_mb",
        "min_run": 3,
        "read_tool": "read_file",
        "path_prefix": "/data/",
    }
    predicate = read_between_constant_field_and_report(base)

    assert (
        predicate(_disk_log(_filling_polls(), reads=[(6, "/data/journal.log")], report_cycle=7))
        is True
    )
    # boundaries: a read exactly at onset or at the report cycle still counts
    assert (
        predicate(_disk_log(_filling_polls(), reads=[(5, "/data/journal.log")], report_cycle=7))
        is True
    )
    assert (
        predicate(_disk_log(_filling_polls(), reads=[(7, "/data/journal.log")], report_cycle=7))
        is True
    )
    assert (
        predicate(_disk_log(_filling_polls(), reads=[(4, "/data/journal.log")], report_cycle=7))
        is False
    )
    assert (
        predicate(_disk_log(_filling_polls(), reads=[(8, "/data/journal.log")], report_cycle=7))
        is False
    )
    assert (
        predicate(_disk_log(_filling_polls(), reads=[(6, "/etc/hosts")], report_cycle=7)) is False
    )


def test_read_between_constant_field_and_report_vacuous_cases() -> None:
    base: PredicateParams = {
        "tool": "run_command",
        "field": "free_mb",
        "min_run": 3,
        "read_tool": "read_file",
        "path_prefix": "/data/",
    }
    predicate = read_between_constant_field_and_report(base)
    advancing = [(cycle + 1, {"free_mb": 100 - cycle}) for cycle in range(6)]

    assert (
        predicate(_disk_log(advancing, reads=[(6, "/data/journal.log")], report_cycle=7)) is False
    )
    assert predicate(_disk_log(_filling_polls(), reads=[(6, "/data/journal.log")])) is False
    assert predicate(EventLog()) is False


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"field": "free_mb", "min_run": 3, "read_tool": "read_file", "path_prefix": "/data/"},
        {"tool": "run_command", "min_run": 3, "read_tool": "read_file", "path_prefix": "/data/"},
        {
            "tool": "run_command",
            "field": "free_mb",
            "read_tool": "read_file",
            "path_prefix": "/data/",
        },
        {"tool": "run_command", "field": "free_mb", "min_run": 3, "path_prefix": "/data/"},
        {"tool": "run_command", "field": "free_mb", "min_run": 3, "read_tool": "read_file"},
        {
            "tool": "run_command",
            "field": "free_mb",
            "min_run": 1,
            "read_tool": "read_file",
            "path_prefix": "/data/",
        },
    ],
)
def test_read_between_constant_field_and_report_rejects_bad_params(
    params: PredicateParams,
) -> None:
    with pytest.raises(ValueError):
        read_between_constant_field_and_report(params)


def test_wave1_all_seventeen_factories_registered() -> None:
    expected = {
        "appends_after_first_post_degradation_readback",
        "capped_search_occurred",
        "command_call_count_at_least",
        "compaction_dropped_seq",
        "distinct_paths_at_least",
        "event_type_absent",
        "first_violation_after_compaction",
        "json_field_threshold_run",
        "post_cap_verification",
        "read_between_constant_field_and_report",
        "readback_after_degradation",
        "report_contains_search_total",
        "report_within_cycles_of_field_threshold",
        "runtime_degradation_present",
        "stale_pins_resolved",
        "tool_calls_since_field_threshold",
        "writes_under_prefix",
    }

    assert expected <= set(REGISTRY)


def test_report_contains_search_total_control_mode_uses_first_search() -> None:
    # control semantics: no capped search exists; the figure is the first
    # search's full total
    log = EventLog()
    t = 0
    for index, (total, capped) in enumerate([(12, False), (30, True)]):
        log.append(
            EventType.TOOL_CALL,
            1,
            t,
            {"call_id": f"c{index}", "tool": "search_code", "args": {"pattern": "x"}},
        )
        t += 1
        log.append(
            EventType.TOOL_RESULT,
            1,
            t,
            {
                "call_id": f"c{index}",
                "output": [["a.py", index + 1]],
                "gt": {"total_matches": total, "returned_matches": min(total, 6), "capped": capped},
            },
        )
        t += 1
    log.append(EventType.REPORT, 2, t, {"report": "audit certificate: exact total 12"})
    predicate = report_contains_search_total({"tool": "search_code", "require_capped": False})
    assert predicate(log) is True
    strict = report_contains_search_total({"tool": "search_code"})
    # default cites the capped total (30); the certificate only carries 12
    assert strict(log) is False


def test_report_contains_search_total_rejects_bad_require_capped() -> None:
    with pytest.raises(ValueError):
        report_contains_search_total({"tool": "search_code", "require_capped": "yes"})
