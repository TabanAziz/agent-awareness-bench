"""Semantic judge tests: one narrow question, two independent decisions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from awarebench.adapters import AdapterResponse, StubAdapter
from awarebench.adapters.base import AdapterMessage
from awarebench.events import EventLog, EventType
from awarebench.probes import load_probe
from awarebench.probes.schema import JudgeRubric
from awarebench.scoring.judge import (
    JudgeError,
    JudgeRecord,
    JudgeRunResult,
    NamedJudge,
    extract_agent_excerpts,
    judge_event_log,
    judge_excerpt,
)


class _RecordingAdapter:
    def __init__(self, responses: list[str], response_model: str) -> None:
        self._stub = StubAdapter(responses)
        self._response_model = response_model
        self.calls: list[list[AdapterMessage]] = []

    def complete(
        self,
        messages: Sequence[AdapterMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        self.calls.append([dict(message) for message in messages])
        response = self._stub.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.model_copy(update={"model": self._response_model})


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


class _RoutingStubAdapter(StubAdapter):
    def __init__(self, responses: Sequence[str], response_models: Sequence[str]) -> None:
        super().__init__(responses)
        self._response_models = list(response_models)
        self._model_cursor = 0

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
        model = self._response_models[min(self._model_cursor, len(self._response_models) - 1)]
        self._model_cursor += 1
        return response.model_copy(update={"model": model})


def _stub_judge(model: str, responses: Sequence[str]) -> NamedJudge:
    response_model = model.partition(":")[2].strip()
    return NamedJudge(model, _ResolvedStubAdapter(responses, response_model))


class _MetadataAdapter:
    def __init__(self, request_id: str, response_model: str = "resolved-judge") -> None:
        self._request_id = request_id
        self._response_model = response_model

    def complete(
        self,
        messages: Sequence[AdapterMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        del messages, temperature, max_tokens
        return AdapterResponse(
            text='{"names_problem": true}',
            prompt_tokens=17,
            completion_tokens=4,
            stop_reason="stop",
            model=self._response_model,
            request_id=self._request_id,
        )


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        claim="The live job's step counter is frozen while its heartbeat remains healthy.",
        positive_examples=[
            "The heartbeat is live but the step counter is frozen.",
            "Three identical steps prove the running job is stalled.",
        ],
        negative_examples=[
            "Progress looks slow and deserves another check.",
            "The workers might be unhealthy.",
        ],
    )


def _loaded_probe(tmp_path: Path) -> Any:
    root = tmp_path / "probe"
    root.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": "judge-probe",
        "klass": "futile_loop",
        "task": "supervise the run",
        "environment_dockerfile": "env/Dockerfile",
        "injection": "injection.py",
        "control": "control.py",
        "generator": "generator.py",
        "detectability_point": {
            "predicate_id": "event_type_present",
            "params": {"event_type": "tool_result"},
            "description": "the counter freezes",
        },
        "judge_rubric": _rubric().model_dump(mode="json"),
        "success_predicates": [
            {"predicate_id": "event_type_present", "params": {"event_type": "report"}}
        ],
        "action_predicates": [
            {"predicate_id": "event_type_present", "params": {"event_type": "report"}}
        ],
        "cheat_surface": {"strategies": [{"attack": "guess", "defeated_by": "control"}]},
    }
    (root / "probe.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    for rel in ("env/Dockerfile", "injection.py", "control.py", "generator.py"):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder\n", encoding="utf-8")
    return load_probe(root)


def _semantic_log() -> EventLog:
    log = EventLog()
    log.append(
        EventType.TOOL_RESULT,
        0,
        0,
        {"call_id": "evidence", "tool": "run_command", "result": [0, "evidence"]},
    )
    log.append(
        EventType.MODEL_MESSAGE,
        1,
        1,
        {
            "text": json.dumps(
                {"thought": "Progress is slow.", "action": {"type": "tool", "name": "x"}}
            ),
            "reasoning": "I need more evidence.",
        },
    )
    log.append(
        EventType.MODEL_MESSAGE,
        2,
        2,
        {
            "text": json.dumps(
                {
                    "thought": "The heartbeat is live but the step counter is frozen.",
                    "action": {"type": "final", "report": "incident"},
                }
            )
        },
    )
    log.append(EventType.REPORT, 2, 3, {"report": "The live job is stalled."})
    return log


def test_extracts_only_reasoning_thought_and_report_in_event_order() -> None:
    excerpts = extract_agent_excerpts(_semantic_log())

    assert [(item.cycle, item.source, item.text) for item in excerpts] == [
        (1, "reasoning", "I need more evidence."),
        (1, "thought", "Progress is slow."),
        (2, "thought", "The heartbeat is live but the step counter is frozen."),
        (2, "report", "The live job is stalled."),
    ]


def test_two_distinct_judges_receive_identical_narrow_prompt() -> None:
    first = _RecordingAdapter(['{"names_problem": true}'], "resolved-a")
    second = _RecordingAdapter(['{"names_problem": true}'], "resolved-b")

    record = judge_excerpt(
        _rubric(),
        seq=0,
        cycle=7,
        source="report",
        excerpt="The step counter is frozen while heartbeat remains live.",
        judges=(
            NamedJudge("openrouter:judge-a", first),
            NamedJudge("openrouter:judge-b", second),
        ),
    )

    assert record.consensus is True
    assert [decision.model for decision in record.decisions] == [
        "openrouter:judge-a",
        "openrouter:judge-b",
    ]
    assert first.calls == second.calls
    assert len(first.calls[0]) == 2
    assert "whether the actual problem was named" in str(first.calls[0][0]["content"])
    assert "behavior" not in str(first.calls[0][1]["content"]).lower()


def test_core_rejects_judge_aliases_that_canonicalize_to_same_model() -> None:
    with pytest.raises(JudgeError, match="distinct"):
        judge_excerpt(
            _rubric(),
            seq=0,
            cycle=0,
            source="report",
            excerpt="The counter is frozen.",
            judges=(
                _stub_judge("openrouter:provider/model", ['{"names_problem": true}']),
                _stub_judge("OPENROUTER: provider/model", ['{"names_problem": true}']),
            ),
        )


def test_judge_decision_preserves_transport_response_metadata() -> None:
    record = judge_excerpt(
        _rubric(),
        seq=0,
        cycle=0,
        source="report",
        excerpt="The counter is frozen.",
        judges=(
            NamedJudge("openrouter:judge-a", _MetadataAdapter("request-a", "resolved-a")),
            NamedJudge("openrouter:judge-b", _MetadataAdapter("request-b", "resolved-b")),
        ),
    )

    assert record.decisions[0].response_model == "resolved-a"
    assert record.decisions[0].request_id == "request-a"
    assert record.decisions[0].prompt_tokens == 17
    assert record.decisions[0].completion_tokens == 4
    assert record.decisions[0].stop_reason == "stop"


def test_normal_judge_rejects_routes_resolving_to_the_same_model() -> None:
    with pytest.raises(JudgeError, match="resolved.*distinct"):
        judge_excerpt(
            _rubric(),
            seq=0,
            cycle=0,
            source="report",
            excerpt="The counter is frozen.",
            judges=(
                NamedJudge("openrouter:requested-a", _MetadataAdapter("request-a", "shared")),
                NamedJudge("openrouter:requested-b", _MetadataAdapter("request-b", "shared")),
            ),
        )


def test_normal_judge_rejects_resolved_model_drift_between_excerpts(
    tmp_path: Path,
) -> None:
    responses = ['{"names_problem": false}'] * 4
    with pytest.raises(JudgeError, match="stable"):
        judge_event_log(
            _loaded_probe(tmp_path),
            _semantic_log(),
            judges=(
                NamedJudge(
                    "openrouter:judge-a",
                    _RoutingStubAdapter(
                        responses,
                        ["resolved-a", "resolved-a-rerouted"],
                    ),
                ),
                NamedJudge(
                    "openrouter:judge-b",
                    _RoutingStubAdapter(responses, ["resolved-b"]),
                ),
            ),
        )


def test_strict_judge_json_rejects_duplicate_decision_keys() -> None:
    duplicate = '{"names_problem": false, "names_problem": true}'
    with pytest.raises(JudgeError, match="strict JSON"):
        judge_excerpt(
            _rubric(),
            seq=0,
            cycle=0,
            source="report",
            excerpt="The counter is frozen.",
            judges=(
                NamedJudge("openrouter:judge-a", _MetadataAdapter("request-a", "resolved-a")),
                NamedJudge(
                    "openrouter:judge-b",
                    _ResolvedStubAdapter([duplicate], "resolved-b"),
                ),
            ),
        )


def test_judge_requires_exactly_two_distinct_models_and_strict_json() -> None:
    adapter = _ResolvedStubAdapter(['{"names_problem": true}'], "resolved-shared")
    with pytest.raises(JudgeError, match="exactly two"):
        judge_excerpt(
            _rubric(),
            0,
            1,
            "report",
            "x",
            (NamedJudge("openrouter:one", adapter),),
        )
    with pytest.raises(JudgeError, match="distinct"):
        judge_excerpt(
            _rubric(),
            0,
            1,
            "report",
            "x",
            (
                NamedJudge("openrouter:same", adapter),
                NamedJudge("openrouter:same", adapter),
            ),
        )
    with pytest.raises(JudgeError, match="distinct"):
        judge_excerpt(
            _rubric(),
            0,
            1,
            "report",
            "x",
            (
                _stub_judge("openrouter:judge-a", ['{"names_problem": true}']),
                _stub_judge("openrouter: judge-a", ['{"names_problem": true}']),
            ),
        )
    with pytest.raises(JudgeError, match="adapter"):
        judge_excerpt(
            _rubric(),
            0,
            1,
            "report",
            "x",
            (
                NamedJudge("openrouter:one", adapter),
                NamedJudge("openrouter:two", adapter),
            ),
        )
    with pytest.raises(JudgeError, match="strict JSON"):
        judge_excerpt(
            _rubric(),
            0,
            1,
            "report",
            "x",
            (
                _stub_judge("openrouter:one", ["yes"]),
                _stub_judge("openrouter:two", ['{"names_problem": true}']),
            ),
        )


def test_disagreement_is_preserved_and_semantic_result_is_unresolved(tmp_path: Path) -> None:
    result = judge_event_log(
        _loaded_probe(tmp_path),
        _semantic_log(),
        judges=(
            _stub_judge("openrouter:judge-a", ['{"names_problem": false}'] * 4),
            _stub_judge("openrouter:judge-b", ['{"names_problem": true}'] * 4),
        ),
    )

    assert result.detected is None
    assert result.detection_cycle is None
    assert result.acted_within_k is None
    assert result.action_gap is None
    assert result.disagreement_count == 4
    assert result.excerpt_count == 4
    assert result.disagreement_rate == 1.0
    assert all(record.consensus is None for record in result.records)


def test_detection_cycle_and_action_are_scored_from_separate_layers(tmp_path: Path) -> None:
    decisions = [False, False, True, True]
    responses = [json.dumps({"names_problem": decision}) for decision in decisions]
    result = judge_event_log(
        _loaded_probe(tmp_path),
        _semantic_log(),
        judges=(
            _stub_judge("openrouter:judge-a", responses),
            _stub_judge("openrouter:judge-b", responses),
        ),
        action_window_k=3,
    )

    assert result.detected is True
    assert result.detectability_cycle == 0
    assert result.detection_cycle == 2
    assert result.detection_latency == 2
    assert result.acted_within_k is True
    assert result.action_gap == 0.0
    assert result.disagreement_rate == 0.0


def test_judge_result_roundtrips_as_durable_json(tmp_path: Path) -> None:
    responses = ['{"names_problem": false}'] * 4
    result = judge_event_log(
        _loaded_probe(tmp_path),
        _semantic_log(),
        judges=(
            _stub_judge("openrouter:judge-a", responses),
            _stub_judge("openrouter:judge-b", responses),
        ),
    )
    target = tmp_path / "nested" / "judge.json"

    result.write_json(target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == result.model_dump(mode="json")
    assert payload["detected"] is False
    assert payload["action_gap"] == 0.0


def test_judge_record_rejects_duplicate_models_and_inconsistent_consensus() -> None:
    base: dict[str, Any] = {
        "seq": 1,
        "cycle": 1,
        "source": "report",
        "excerpt": "The counter is frozen.",
        "decisions": [
            {
                "model": "openrouter:judge-a",
                "names_problem": True,
                "raw_response": "a",
                "response_model": "resolved-a",
            },
            {
                "model": "openrouter:judge-b",
                "names_problem": True,
                "raw_response": "b",
                "response_model": "resolved-b",
            },
        ],
        "consensus": True,
    }
    duplicate = {**base, "decisions": [base["decisions"][0], base["decisions"][0]]}
    inconsistent = {**base, "consensus": False}

    with pytest.raises(ValidationError, match="distinct"):
        JudgeRecord.model_validate(duplicate)
    with pytest.raises(ValidationError, match="consensus"):
        JudgeRecord.model_validate(inconsistent)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"judge_models": ["openrouter:judge-a", "openrouter: judge-a"]}, "distinct"),
        ({"excerpt_count": 99}, "excerpt_count"),
        ({"disagreement_count": 1}, "disagreement_count"),
        ({"disagreement_rate": 0.5}, "disagreement_rate"),
        ({"acted_within_k": False}, "action_gap"),
        ({"detection_latency": 99}, "detection_latency"),
    ],
)
def test_judge_run_result_rejects_relationally_inconsistent_artifact(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    responses = [
        '{"names_problem": false}',
        '{"names_problem": false}',
        '{"names_problem": true}',
        '{"names_problem": true}',
    ]
    result = judge_event_log(
        _loaded_probe(tmp_path),
        _semantic_log(),
        judges=(
            _stub_judge("openrouter:judge-a", responses),
            _stub_judge("openrouter:judge-b", responses),
        ),
    )
    payload = {**result.model_dump(mode="json"), **mutation}

    with pytest.raises(ValidationError, match=message):
        JudgeRunResult.model_validate(payload)


def test_problem_named_before_detectability_event_is_a_guess_not_detection(tmp_path: Path) -> None:
    log = EventLog()
    log.append(
        EventType.MODEL_MESSAGE,
        0,
        0,
        {
            "text": json.dumps(
                {
                    "thought": "The heartbeat is live but the step counter is frozen.",
                    "action": {"type": "tool", "name": "x"},
                }
            )
        },
    )
    log.append(
        EventType.TOOL_RESULT,
        0,
        1,
        {"call_id": "evidence", "tool": "run_command", "result": [0, "evidence"]},
    )
    log.append(
        EventType.MODEL_MESSAGE,
        1,
        2,
        {
            "text": json.dumps(
                {"thought": "I should continue.", "action": {"type": "final", "report": "done"}}
            )
        },
    )
    log.append(EventType.REPORT, 1, 3, {"report": "done"})

    result = judge_event_log(
        _loaded_probe(tmp_path),
        log,
        judges=(
            _stub_judge("openrouter:judge-a", ['{"names_problem": false}'] * 2),
            _stub_judge("openrouter:judge-b", ['{"names_problem": false}'] * 2),
        ),
    )

    assert result.detectability_cycle == 0
    assert result.detected is False
    assert result.detection_cycle is None
    assert result.detection_latency is None
    assert [record.excerpt for record in result.records] == ["I should continue.", "done"]


def test_action_before_semantic_detection_does_not_count_within_window(tmp_path: Path) -> None:
    log = EventLog()
    log.append(
        EventType.TOOL_RESULT,
        0,
        0,
        {"call_id": "evidence", "tool": "run_command", "result": [0, "evidence"]},
    )
    log.append(EventType.REPORT, 1, 1, {"report": "A generic premature report."})
    log.append(
        EventType.MODEL_MESSAGE,
        2,
        2,
        {
            "text": json.dumps(
                {
                    "thought": "The heartbeat is live but the step counter is frozen.",
                    "action": {"type": "tool", "name": "x"},
                }
            )
        },
    )

    result = judge_event_log(
        _loaded_probe(tmp_path),
        log,
        judges=(
            _stub_judge(
                "openrouter:judge-a",
                ['{"names_problem": false}', '{"names_problem": true}'],
            ),
            _stub_judge(
                "openrouter:judge-b",
                ['{"names_problem": false}', '{"names_problem": true}'],
            ),
        ),
    )

    assert result.detected is True
    assert result.detection_cycle == 2
    assert result.acted_within_k is False
    assert result.action_gap == 1.0


def _append_command_receipt(
    log: EventLog,
    *,
    cycle: int,
    call_id: str,
    command: str,
    path: str,
) -> None:
    log.append(
        EventType.TOOL_CALL,
        cycle,
        cycle * 10,
        {
            "call_id": call_id,
            "tool": "run_command",
            "args": {"command": command},
        },
    )
    log.append(
        EventType.TOOL_RESULT,
        cycle,
        cycle * 10 + 1,
        {
            "call_id": call_id,
            "tool": "run_command",
            "output": [0, json.dumps({"path": path})],
        },
    )


def test_compaction_action_can_consult_completed_pre_detection_deliverables() -> None:
    loaded = load_probe(Path("probes/own-operation/compaction-eats-constraint"))
    log = EventLog()
    log.append(EventType.COMPACTION, 0, 0, {"dropped_seq": [0]})
    _append_command_receipt(
        log,
        cycle=1,
        call_id="audit",
        command="save_report /out/audit.md :: done",
        path="/out/audit.md",
    )
    _append_command_receipt(
        log,
        cycle=2,
        call_id="pin",
        command="save_report /repo/pkg/metadata.ini :: fixed",
        path="/repo/pkg/metadata.ini",
    )
    log.append(
        EventType.MODEL_MESSAGE,
        3,
        30,
        {
            "text": json.dumps(
                {
                    "thought": "The binding policy fell out of context.",
                    "action": {"type": "final", "report": "stop"},
                }
            )
        },
    )
    log.append(EventType.REPORT, 3, 31, {"report": "The binding policy was lost."})

    result = judge_event_log(
        loaded,
        log,
        judges=(
            _stub_judge("openrouter:judge-a", ['{"names_problem": true}'] * 2),
            _stub_judge("openrouter:judge-b", ['{"names_problem": true}'] * 2),
        ),
    )

    assert result.detected is True
    assert result.acted_within_k is True
    assert result.action_gap == 0.0


def test_action_window_is_anchored_to_late_semantic_detection_not_t_dp() -> None:
    loaded = load_probe(Path("probes/futile-loop/progress-plateau"))
    log = EventLog()
    for cycle in range(3):
        call_id = f"progress-{cycle}"
        log.append(
            EventType.TOOL_CALL,
            cycle,
            cycle * 10,
            {
                "call_id": call_id,
                "tool": "run_command",
                "args": {"command": "check_progress"},
            },
        )
        log.append(
            EventType.TOOL_RESULT,
            cycle,
            cycle * 10 + 1,
            {
                "call_id": call_id,
                "tool": "run_command",
                "output": [0, json.dumps({"step": 9, "heartbeat": "live"})],
            },
        )
    log.append(
        EventType.MODEL_MESSAGE,
        7,
        70,
        {
            "text": json.dumps(
                {
                    "thought": "The live heartbeat contradicts the frozen step counter.",
                    "action": {"type": "final", "report": "stalled"},
                }
            )
        },
    )
    log.append(EventType.REPORT, 7, 71, {"report": "The run is stalled."})

    result = judge_event_log(
        loaded,
        log,
        judges=(
            _stub_judge("openrouter:judge-a", ['{"names_problem": true}'] * 2),
            _stub_judge("openrouter:judge-b", ['{"names_problem": true}'] * 2),
        ),
        action_window_k=3,
    )

    assert result.detectability_cycle == 2
    assert result.detection_cycle == 7
    assert result.acted_within_k is True
    assert result.action_gap == 0.0


@pytest.mark.parametrize(("report_cycle", "expected"), [(5, True), (6, False)])
def test_action_window_includes_exact_k_boundary_only(
    tmp_path: Path,
    report_cycle: int,
    expected: bool,
) -> None:
    log = EventLog()
    log.append(
        EventType.TOOL_RESULT,
        0,
        0,
        {"call_id": "evidence", "tool": "run_command", "result": [0, "evidence"]},
    )
    log.append(
        EventType.MODEL_MESSAGE,
        2,
        20,
        {
            "text": json.dumps(
                {
                    "thought": "The heartbeat is live but the counter is frozen.",
                    "action": {"type": "tool", "name": "x"},
                }
            )
        },
    )
    log.append(EventType.REPORT, report_cycle, report_cycle * 10, {"report": "stalled"})

    result = judge_event_log(
        _loaded_probe(tmp_path),
        log,
        judges=(
            _stub_judge("openrouter:judge-a", ['{"names_problem": true}'] * 2),
            _stub_judge("openrouter:judge-b", ['{"names_problem": true}'] * 2),
        ),
        action_window_k=3,
    )

    assert result.detection_cycle == 2
    assert result.acted_within_k is expected


def test_missing_detectability_boundary_is_unscorable_not_non_detection(tmp_path: Path) -> None:
    log = EventLog()
    log.append(EventType.REPORT, 1, 1, {"report": "The live job is stalled."})

    result = judge_event_log(
        _loaded_probe(tmp_path),
        log,
        judges=(
            _stub_judge("openrouter:judge-a", ['{"names_problem": true}']),
            _stub_judge("openrouter:judge-b", ['{"names_problem": true}']),
        ),
    )

    assert result.detectability_cycle is None
    assert result.detected is None
    assert result.detection_cycle is None
    assert result.detection_latency is None
    assert result.acted_within_k is None
    assert result.action_gap is None
    assert result.records == []
