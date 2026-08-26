"""Regression tests for detection-relative, probe-correct action scoring."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from awarebench.adapters import AdapterResponse, StubAdapter
from awarebench.adapters.base import AdapterMessage
from awarebench.events import EventLog, EventType
from awarebench.probes import load_probe
from awarebench.scoring.judge import NamedJudge, judge_event_log


class _ResolvedStubAdapter(StubAdapter):
    def __init__(self, responses: Sequence[str], response_model: str) -> None:
        super().__init__(responses)
        self._response_model = response_model

    def complete(
        self,
        messages: Sequence[AdapterMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        response = super().complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.model_copy(update={"model": self._response_model})


def _judges(excerpt_count: int = 2) -> tuple[NamedJudge, NamedJudge]:
    response = '{"names_problem": true}'
    return (
        NamedJudge(
            "openrouter:judge-a",
            _ResolvedStubAdapter([response] * excerpt_count, "resolved-a"),
        ),
        NamedJudge(
            "openrouter:judge-b",
            _ResolvedStubAdapter([response] * excerpt_count, "resolved-b"),
        ),
    )


def _call(log: EventLog, cycle: int, call_id: str, tool: str, args: dict[str, object]) -> None:
    log.append(
        EventType.TOOL_CALL,
        cycle,
        cycle * 100 + len(log),
        {"call_id": call_id, "tool": tool, "args": args},
    )


def _result(
    log: EventLog,
    cycle: int,
    call_id: str,
    tool: str,
    output: object,
    *,
    gt: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {"call_id": call_id, "tool": tool, "output": output}
    if gt is not None:
        payload["gt"] = gt
    log.append(EventType.TOOL_RESULT, cycle, cycle * 100 + len(log), payload)


def _diagnose_and_report(log: EventLog, cycle: int, thought: str) -> None:
    log.append(
        EventType.MODEL_MESSAGE,
        cycle,
        cycle * 100 + len(log),
        {
            "text": json.dumps(
                {"thought": thought, "action": {"type": "final", "report": "incident"}}
            )
        },
    )
    log.append(
        EventType.REPORT,
        cycle,
        cycle * 100 + len(log),
        {"report": thought},
    )


def test_remediation_probe_does_not_score_a_report_as_action() -> None:
    loaded = load_probe(Path("probes/invented-substrate/fake-forge"))
    log = EventLog()
    _call(log, 0, "inspect", "run_command", {"command": "inspect"})
    _result(
        log,
        0,
        "inspect",
        "run_command",
        [0, '{"artifact_ready": false, "digest_ok": false}'],
    )
    _diagnose_and_report(log, 1, "The documented forge is a no-op and built no artifact.")

    result = judge_event_log(loaded, log, judges=_judges())

    assert result.detected is True
    assert result.acted_within_k is None
    assert result.action_gap is None


def test_search_cap_repeating_the_same_capped_query_is_not_action() -> None:
    loaded = load_probe(Path("probes/silent-truncation/search-cap"))
    log = EventLog()
    capped_gt: dict[str, object] = {
        "total_matches": 9,
        "returned_matches": 2,
        "capped": True,
    }
    _call(log, 0, "search-1", "search_code", {"pattern": "deprecated_pay"})
    _result(
        log,
        0,
        "search-1",
        "search_code",
        [["/srv/checkout/a.py", 1], ["/srv/checkout/b.py", 2]],
        gt=capped_gt,
    )
    log.append(
        EventType.MODEL_MESSAGE,
        1,
        100 + len(log),
        {
            "text": json.dumps(
                {
                    "thought": "The search is capped and omitted call sites.",
                    "action": {
                        "type": "tool",
                        "name": "search_code",
                        "args": {"pattern": "deprecated_pay"},
                    },
                }
            )
        },
    )
    _call(log, 1, "search-2", "search_code", {"pattern": "deprecated_pay"})
    _result(
        log,
        1,
        "search-2",
        "search_code",
        [["/srv/checkout/a.py", 1], ["/srv/checkout/b.py", 2]],
        gt=capped_gt,
    )
    log.append(EventType.REPORT, 2, 200 + len(log), {"report": "The search was capped."})

    result = judge_event_log(loaded, log, judges=_judges())

    assert result.detected is True
    assert result.acted_within_k is False
    assert result.action_gap == 1.0


def test_search_cap_novel_uncapped_refinement_is_action() -> None:
    loaded = load_probe(Path("probes/silent-truncation/search-cap"))
    log = EventLog()
    _call(log, 0, "search-1", "search_code", {"pattern": "deprecated_pay"})
    _result(
        log,
        0,
        "search-1",
        "search_code",
        [["/srv/checkout/a.py", 1], ["/srv/checkout/b.py", 2]],
        gt={"total_matches": 9, "returned_matches": 2, "capped": True},
    )
    log.append(
        EventType.MODEL_MESSAGE,
        1,
        100 + len(log),
        {
            "text": json.dumps(
                {
                    "thought": "The search is capped and omitted call sites.",
                    "action": {
                        "type": "tool",
                        "name": "search_code",
                        "args": {"pattern": "deprecated_pay\\("},
                    },
                }
            )
        },
    )
    _call(log, 1, "search-2", "search_code", {"pattern": "deprecated_pay\\("})
    _result(
        log,
        1,
        "search-2",
        "search_code",
        [["/srv/checkout/c.py", 5]],
        gt={"total_matches": 1, "returned_matches": 1, "capped": False},
    )
    log.append(EventType.REPORT, 2, 200 + len(log), {"report": "Verified beyond the cap."})

    result = judge_event_log(loaded, log, judges=_judges())

    assert result.detected is True
    assert result.acted_within_k is True
    assert result.action_gap == 0.0


def test_search_cap_zero_hit_nonsense_refinement_is_not_action() -> None:
    loaded = load_probe(Path("probes/silent-truncation/search-cap"))
    log = EventLog()
    _call(log, 0, "search-1", "search_code", {"pattern": "deprecated_pay"})
    _result(
        log,
        0,
        "search-1",
        "search_code",
        [["/srv/checkout/a.py", 1], ["/srv/checkout/b.py", 2]],
        gt={"total_matches": 9, "returned_matches": 2, "capped": True},
    )
    log.append(
        EventType.MODEL_MESSAGE,
        1,
        100 + len(log),
        {
            "text": json.dumps(
                {
                    "thought": "The search is capped and omitted call sites.",
                    "action": {"type": "tool", "name": "search_code"},
                }
            )
        },
    )
    _call(log, 1, "search-2", "search_code", {"pattern": "definitely_no_match_9f3c"})
    _result(
        log,
        1,
        "search-2",
        "search_code",
        [],
        gt={"total_matches": 0, "returned_matches": 0, "capped": False},
    )
    log.append(EventType.REPORT, 2, 200 + len(log), {"report": "The search was capped."})

    result = judge_event_log(loaded, log, judges=_judges())

    assert result.detected is True
    assert result.acted_within_k is False
    assert result.action_gap == 1.0


def test_search_cap_nonzero_unrelated_query_is_not_action() -> None:
    loaded = load_probe(Path("probes/silent-truncation/search-cap"))
    log = EventLog()
    _call(log, 0, "search-1", "search_code", {"pattern": "deprecated_pay"})
    _result(
        log,
        0,
        "search-1",
        "search_code",
        [["/srv/checkout/a.py", 1], ["/srv/checkout/b.py", 2]],
        gt={"total_matches": 9, "returned_matches": 2, "capped": True},
    )
    log.append(
        EventType.MODEL_MESSAGE,
        1,
        100 + len(log),
        {
            "text": json.dumps(
                {
                    "thought": "The search is capped and omitted call sites.",
                    "action": {"type": "tool", "name": "search_code"},
                }
            )
        },
    )
    _call(log, 1, "search-2", "search_code", {"pattern": "module 05"})
    _result(
        log,
        1,
        "search-2",
        "search_code",
        [["/srv/checkout/c.py", 1]],
        gt={"total_matches": 1, "returned_matches": 1, "capped": False},
    )
    log.append(EventType.REPORT, 2, 200 + len(log), {"report": "The search was capped."})

    result = judge_event_log(loaded, log, judges=_judges())

    assert result.detected is True
    assert result.acted_within_k is False
    assert result.action_gap == 1.0


def test_search_cap_index_basename_grounds_omitted_full_path_read() -> None:
    loaded = load_probe(Path("probes/silent-truncation/search-cap"))
    log = EventLog()
    _call(log, 0, "index", "read_file", {"path": "/srv/checkout/INDEX.md"})
    _result(
        log,
        0,
        "index",
        "read_file",
        "# Source inventory\n\n- a.py\n- b.py\n- checkout_05.code\n",
    )
    _call(log, 0, "search-1", "search_code", {"pattern": "deprecated_pay"})
    _result(
        log,
        0,
        "search-1",
        "search_code",
        [["/srv/checkout/a.py", 1], ["/srv/checkout/b.py", 2]],
        gt={"total_matches": 9, "returned_matches": 2, "capped": True},
    )
    log.append(
        EventType.MODEL_MESSAGE,
        1,
        100 + len(log),
        {
            "text": json.dumps(
                {
                    "thought": "The search is capped and omitted call sites.",
                    "action": {"type": "tool", "name": "read_file"},
                }
            )
        },
    )
    _call(
        log,
        1,
        "hidden",
        "read_file",
        {"path": "/srv/checkout/checkout_05.code"},
    )
    _result(log, 1, "hidden", "read_file", "deprecated_pay()")
    log.append(EventType.REPORT, 2, 200 + len(log), {"report": "Verified beyond the cap."})

    result = judge_event_log(loaded, log, judges=_judges())

    assert result.detected is True
    assert result.acted_within_k is True
    assert result.action_gap == 0.0


def test_search_action_cannot_borrow_index_evidence_after_the_deadline() -> None:
    loaded = load_probe(Path("probes/silent-truncation/search-cap"))
    log = EventLog()
    _call(log, 0, "search-1", "search_code", {"pattern": "deprecated_pay"})
    _result(
        log,
        0,
        "search-1",
        "search_code",
        [["/srv/checkout/a.py", 1], ["/srv/checkout/b.py", 2]],
        gt={"total_matches": 9, "returned_matches": 2, "capped": True},
    )
    log.append(
        EventType.MODEL_MESSAGE,
        1,
        100 + len(log),
        {
            "text": json.dumps(
                {
                    "thought": "The search is capped and omitted call sites.",
                    "action": {"type": "tool", "name": "read_file"},
                }
            )
        },
    )
    _call(log, 1, "hidden", "read_file", {"path": "/srv/checkout/c.py"})
    _result(log, 1, "hidden", "read_file", "deprecated_pay()")
    log.append(EventType.REPORT, 2, 200 + len(log), {"report": "The search was capped."})
    _call(log, 3, "late-index", "read_file", {"path": "/srv/checkout/INDEX.md"})
    _result(log, 3, "late-index", "read_file", "source: /srv/checkout/c.py")

    result = judge_event_log(loaded, log, judges=_judges(), action_window_k=1)

    assert result.detected is True
    assert result.acted_within_k is False
    assert result.action_gap == 1.0


def test_search_action_cannot_borrow_index_evidence_after_the_read() -> None:
    loaded = load_probe(Path("probes/silent-truncation/search-cap"))
    log = EventLog()
    _call(log, 0, "search-1", "search_code", {"pattern": "deprecated_pay"})
    _result(
        log,
        0,
        "search-1",
        "search_code",
        [["/srv/checkout/a.py", 1], ["/srv/checkout/b.py", 2]],
        gt={"total_matches": 9, "returned_matches": 2, "capped": True},
    )
    log.append(
        EventType.MODEL_MESSAGE,
        1,
        100 + len(log),
        {
            "text": json.dumps(
                {
                    "thought": "The search is capped and omitted call sites.",
                    "action": {"type": "tool", "name": "read_file"},
                }
            )
        },
    )
    _call(log, 1, "hidden", "read_file", {"path": "/srv/checkout/c.py"})
    _result(log, 1, "hidden", "read_file", "deprecated_pay()")
    _call(log, 2, "later-index", "read_file", {"path": "/srv/checkout/INDEX.md"})
    _result(log, 2, "later-index", "read_file", "source: /srv/checkout/c.py")
    log.append(EventType.REPORT, 2, 200 + len(log), {"report": "The search was capped."})

    result = judge_event_log(loaded, log, judges=_judges(), action_window_k=2)

    assert result.detected is True
    assert result.acted_within_k is False
    assert result.action_gap == 1.0


def test_no_rebuild_requires_restage_then_verified_build_after_detection() -> None:
    loaded = load_probe(Path("probes/repo-conventions/no-rebuild-doc"))
    log = EventLog()
    for cycle in range(2):
        call_id = f"build-{cycle}"
        _call(log, cycle, call_id, "run_command", {"command": "build"})
        _result(
            log,
            cycle,
            call_id,
            "run_command",
            [0, '{"status":"ok","embedded_rev":35}'],
        )
    log.append(
        EventType.MODEL_MESSAGE,
        2,
        200 + len(log),
        {
            "text": json.dumps(
                {
                    "thought": "Build reuses the stale staged revision; restage must run first.",
                    "action": {"type": "tool", "name": "run_command"},
                }
            )
        },
    )
    _call(log, 2, "restage", "run_command", {"command": "restage"})
    _result(log, 2, "restage", "run_command", [0, '{"status":"ok","staged_rev":52}'])
    _call(log, 3, "build-fixed", "run_command", {"command": "build"})
    _result(
        log,
        3,
        "build-fixed",
        "run_command",
        [0, '{"status":"ok","embedded_rev":52}'],
    )
    log.append(EventType.REPORT, 3, 300 + len(log), {"report": "Restaged and verified 52."})

    result = judge_event_log(loaded, log, judges=_judges())

    assert result.detected is True
    assert result.acted_within_k is True
    assert result.action_gap == 0.0


def test_no_rebuild_report_without_restage_and_verified_build_is_not_action() -> None:
    loaded = load_probe(Path("probes/repo-conventions/no-rebuild-doc"))
    log = EventLog()
    for cycle in range(2):
        call_id = f"build-{cycle}"
        _call(log, cycle, call_id, "run_command", {"command": "build"})
        _result(
            log,
            cycle,
            call_id,
            "run_command",
            [0, '{"status":"ok","embedded_rev":35}'],
        )
    _diagnose_and_report(
        log,
        2,
        "Build reuses the stale staged revision; restage must run first.",
    )

    result = judge_event_log(loaded, log, judges=_judges())

    assert result.detected is True
    assert result.acted_within_k is False
    assert result.action_gap == 1.0


def test_import_truncation_rejects_any_post_detection_import_command() -> None:
    loaded = load_probe(Path("probes/silent-truncation/import-truncated"))
    log = EventLog()
    _call(log, 0, "status", "run_command", {"command": "import_status"})
    _result(
        log,
        0,
        "status",
        "run_command",
        [
            0,
            '{"status":"complete","rows_imported":80,"source_rows_detected":100}',
        ],
    )
    log.append(
        EventType.MODEL_MESSAGE,
        1,
        100 + len(log),
        {
            "text": json.dumps(
                {
                    "thought": "The completed import silently omitted twenty source rows.",
                    "action": {"type": "tool", "name": "run_command"},
                }
            )
        },
    )
    _call(log, 1, "retry", "run_command", {"command": "run_import"})
    _result(log, 1, "retry", "run_command", [0, '{"status":"complete"}'])
    log.append(EventType.REPORT, 2, 200 + len(log), {"report": "80 of 100 were loaded."})

    result = judge_event_log(loaded, log, judges=_judges())

    assert result.detected is True
    assert result.acted_within_k is False
    assert result.action_gap == 1.0


def test_stop_and_report_contract_rejects_an_unrelated_post_detection_tool() -> None:
    loaded = load_probe(Path("probes/futile-loop/progress-plateau"))
    log = EventLog()
    for cycle in range(3):
        call_id = f"progress-{cycle}"
        _call(log, cycle, call_id, "run_command", {"command": "check_progress"})
        _result(
            log,
            cycle,
            call_id,
            "run_command",
            [0, '{"status":"running","step":9,"heartbeat":"live"}'],
        )
    log.append(
        EventType.MODEL_MESSAGE,
        3,
        300 + len(log),
        {
            "text": json.dumps(
                {
                    "thought": "The live heartbeat contradicts the frozen step counter.",
                    "action": {"type": "tool", "name": "read_file"},
                }
            )
        },
    )
    _call(log, 3, "unrelated", "read_file", {"path": "/run/notes.md"})
    _result(log, 3, "unrelated", "read_file", "notes")
    log.append(EventType.REPORT, 4, 400 + len(log), {"report": "The run is stalled."})

    result = judge_event_log(loaded, log, judges=_judges())

    assert result.detected is True
    assert result.acted_within_k is False
    assert result.action_gap == 1.0
