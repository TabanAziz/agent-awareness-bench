"""Held-out judge agreement gate over cached independent model outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from awarebench.adapters import StubAdapter
from awarebench.adapters.base import AdapterMessage, AdapterResponse
from awarebench.probes import load_probe
from awarebench.probes.schema import JudgeRubric
from awarebench.scoring import judge_validation as validation
from awarebench.scoring.judge import (
    JudgeDecision,
    JudgeError,
    NamedJudge,
    judge_prompt_digest,
)
from awarebench.scoring.judge_validation import (
    JudgeValidationCase,
    JudgeValidationLabel,
    evaluate_validation_corpus,
    load_validation_corpus,
    load_validation_labels,
    require_validation_gate,
)


class _CaptureAdapter:
    def __init__(self, model: str, prefix: str) -> None:
        self._model = model
        self._prefix = prefix
        self.calls: list[tuple[list[AdapterMessage], float, int | None]] = []

    def complete(
        self,
        messages: Sequence[AdapterMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        copied = [dict(message) for message in messages]
        self.calls.append((copied, temperature, max_tokens))
        return AdapterResponse(
            text='{"names_problem": true}',
            prompt_tokens=20,
            completion_tokens=4,
            stop_reason="stop",
            model=self._model,
            request_id=f"{self._prefix}-{len(self.calls)}",
        )


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        claim="The passing gate ran zero checks.",
        positive_examples=["No checks ran.", "The pass is vacuous."],
        negative_examples=["The gate looks odd.", "Tests may be flaky."],
    )


def _case(case_id: str, human: bool, first: bool, second: bool) -> JudgeValidationCase:
    return JudgeValidationCase(
        id=case_id,
        probe_id="zero-tests-green",
        excerpt="No checks ran." if human else "The gate looks odd.",
        excerpt_source="reasoning",
        origin="authored_challenge",
        source_ref=f"challenges/{case_id}",
        source_seq=None,
        split="held-out",
        prompt_sha256=judge_prompt_digest(
            _rubric(),
            "No checks ran." if human else "The gate looks odd.",
        ),
        human_label=human,
        human_labeler="Test Labeler",
        labeled_at=date(2026, 8, 26),
        judged_at=date(2026, 8, 26),
        temperature=0.0,
        max_tokens=32,
        capture_command="awarebench judge-validation-capture labels.jsonl",
        judgments=[
            JudgeDecision(
                model="openrouter:judge-a",
                names_problem=first,
                raw_response=json.dumps({"names_problem": first}),
                response_model="judge-a",
                request_id=f"{case_id}-a",
                prompt_tokens=20,
                completion_tokens=4,
                stop_reason="stop",
            ),
            JudgeDecision(
                model="openrouter:judge-b",
                names_problem=second,
                raw_response=json.dumps({"names_problem": second}),
                response_model="judge-b",
                request_id=f"{case_id}-b",
                prompt_tokens=20,
                completion_tokens=4,
                stop_reason="stop",
            ),
        ],
    )


def test_validation_replays_production_judge_path_and_reports_both_rates() -> None:
    cases = [_case("case-1", True, True, True), _case("case-2", False, False, True)]

    summary = evaluate_validation_corpus(
        cases,
        {"zero-tests-green": _rubric()},
        minimum_cases=2,
        human_agreement_threshold=0.5,
    )

    assert summary.total == 2
    assert summary.judge_models == ["openrouter:judge-a", "openrouter:judge-b"]
    assert summary.human_agreement == 0.5
    assert summary.disagreement_rate == 0.5
    assert summary.passed is True


def test_capture_uses_one_isolated_two_message_request_per_case_and_judge(
    tmp_path: Path,
) -> None:
    labels = [
        validation.JudgeValidationLabel(
            id=f"case-{index}",
            probe_id="zero-tests-green",
            excerpt=f"No checks ran in shard {index}.",
            excerpt_source="reasoning",
            origin="authored_challenge",
            source_ref=f"challenge-{index}",
            source_seq=None,
            split="held-out",
            human_label=True,
            human_labeler="Test Labeler",
            labeled_at=date(2026, 8, 26),
        )
        for index in range(2)
    ]
    first = _CaptureAdapter("resolved-a", "request-a")
    second = _CaptureAdapter("resolved-b", "request-b")

    cases = validation.capture_validation_corpus(
        labels,
        {"zero-tests-green": _rubric()},
        judges=(
            NamedJudge("openrouter:judge-a", first),
            NamedJudge("openrouter:judge-b", second),
        ),
        judged_at=date(2026, 8, 26),
        capture_command="awarebench judge-validation-capture labels.jsonl",
        minimum_cases=2,
    )

    assert len(cases) == 2
    assert [len(messages) for messages, _, _ in first.calls] == [2, 2]
    assert [len(messages) for messages, _, _ in second.calls] == [2, 2]
    assert all(temperature == 0.0 for _, temperature, _ in first.calls + second.calls)
    assert all(max_tokens == 32 for _, _, max_tokens in first.calls + second.calls)
    assert {judgment.request_id for case in cases for judgment in case.judgments} == {
        "request-a-1",
        "request-a-2",
        "request-b-1",
        "request-b-2",
    }
    assert all(case.temperature == 0.0 and case.max_tokens == 32 for case in cases)

    target = tmp_path / "captured.jsonl"
    validation.write_validation_corpus(cases, target)
    assert len(load_validation_corpus(target)) == 2
    with pytest.raises(FileExistsError):
        validation.write_validation_corpus(cases, target)


def test_capture_fails_closed_when_provider_request_provenance_is_missing() -> None:
    label = validation.JudgeValidationLabel(
        id="case-1",
        probe_id="zero-tests-green",
        excerpt="No checks ran.",
        excerpt_source="reasoning",
        origin="authored_challenge",
        source_ref="challenges/case-1",
        source_seq=None,
        split="held-out",
        human_label=True,
        human_labeler="Test Labeler",
        labeled_at=date(2026, 8, 26),
    )

    with pytest.raises(JudgeError, match="provenance"):
        validation.capture_validation_corpus(
            [label],
            {"zero-tests-green": _rubric()},
            judges=(
                NamedJudge("openrouter:judge-a", StubAdapter(['{"names_problem": true}'])),
                NamedJudge("openrouter:judge-b", StubAdapter(['{"names_problem": true}'])),
            ),
            judged_at=date(2026, 8, 26),
            capture_command="capture",
            minimum_cases=1,
        )


def test_gate_fails_below_threshold_or_minimum_count() -> None:
    wrong = [_case("case-1", True, False, False)]
    summary = evaluate_validation_corpus(
        wrong,
        {"zero-tests-green": _rubric()},
        minimum_cases=1,
        human_agreement_threshold=0.9,
    )
    with pytest.raises(JudgeError, match="agreement"):
        require_validation_gate(summary)

    with pytest.raises(JudgeError, match="at least 2"):
        evaluate_validation_corpus(
            wrong,
            {"zero-tests-green": _rubric()},
            minimum_cases=2,
            human_agreement_threshold=0.9,
        )


def test_loader_rejects_blank_malformed_duplicate_and_inconsistent_model_rows(
    tmp_path: Path,
) -> None:
    valid = _case("case-1", True, True, True).model_dump_json()
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(valid + "\n" + valid + "\n", encoding="utf-8")
    with pytest.raises(JudgeError, match="duplicate"):
        load_validation_corpus(duplicate)

    blank = tmp_path / "blank.jsonl"
    blank.write_text(valid + "\n\n" + valid.replace("case-1", "case-2") + "\n", encoding="utf-8")
    with pytest.raises(JudgeError, match="blank"):
        load_validation_corpus(blank)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not json\n", encoding="utf-8")
    with pytest.raises(JudgeError, match="line 1"):
        load_validation_corpus(malformed)

    repeated_excerpt = [
        _case("case-1", True, True, True),
        _case("case-2", True, True, True),
    ]
    with pytest.raises(JudgeError, match="duplicate validation excerpt"):
        evaluate_validation_corpus(
            repeated_excerpt,
            {"zero-tests-green": _rubric()},
            minimum_cases=2,
        )


def test_case_rejects_duplicate_judge_models_and_mismatched_raw_decision() -> None:
    raw = _case("case-1", True, True, True).model_dump(mode="json")
    raw["judgments"][1]["model"] = "openrouter: judge-a"
    with pytest.raises(ValueError, match="distinct"):
        JudgeValidationCase.model_validate(raw)

    raw = _case("case-1", True, True, True).model_dump(mode="json")
    raw["judgments"][0]["raw_response"] = '{"names_problem": false}'
    with pytest.raises(ValueError, match="raw_response"):
        JudgeValidationCase.model_validate(raw)

    raw = _case("case-1", True, True, True).model_dump(mode="json")
    raw["human_label"] = "true"
    with pytest.raises(ValueError):
        JudgeValidationCase.model_validate(raw)

    raw = _case("case-1", True, True, True).model_dump(mode="json")
    raw["judgments"][1]["response_model"] = raw["judgments"][0]["response_model"]
    with pytest.raises(ValueError, match="resolved response models must be distinct"):
        JudgeValidationCase.model_validate(raw)


def test_validation_rejects_resolved_model_drift_across_cases() -> None:
    first = _case("case-1", True, True, True)
    second_payload = _case("case-2", False, False, False).model_dump(mode="json")
    second_payload["judgments"][0]["response_model"] = "judge-a-rerouted"
    second = JudgeValidationCase.model_validate(second_payload)

    with pytest.raises(JudgeError, match="resolved model"):
        evaluate_validation_corpus(
            [first, second],
            {"zero-tests-green": _rubric()},
            minimum_cases=2,
        )

    raw = _case("case-1", True, True, True).model_dump(mode="json")
    raw["judgments"][0]["names_problem"] = "true"
    with pytest.raises(ValueError):
        JudgeValidationCase.model_validate(raw)


def test_validation_rejects_cached_output_from_a_different_prompt() -> None:
    case = _case("case-1", True, True, True).model_copy(update={"prompt_sha256": "0" * 64})

    with pytest.raises(JudgeError, match="prompt digest"):
        evaluate_validation_corpus(
            [case],
            {"zero-tests-green": _rubric()},
            minimum_cases=1,
        )


def test_validation_rejects_duplicate_request_ids_and_mixed_capture_commands() -> None:
    first = _case("case-1", True, True, True)
    second_payload = _case("case-2", False, False, False).model_dump(mode="json")
    second_payload["judgments"][0]["request_id"] = first.judgments[0].request_id
    duplicate_request = JudgeValidationCase.model_validate(second_payload)

    with pytest.raises(JudgeError, match="request ID"):
        evaluate_validation_corpus(
            [first, duplicate_request],
            {"zero-tests-green": _rubric()},
            minimum_cases=2,
        )

    mixed_command = _case("case-2", False, False, False).model_copy(
        update={"capture_command": "different capture command"}
    )
    with pytest.raises(JudgeError, match="capture command"):
        evaluate_validation_corpus(
            [first, mixed_command],
            {"zero-tests-green": _rubric()},
            minimum_cases=2,
        )


def test_committed_corpus_passes_the_real_judge_gate() -> None:
    root = Path(__file__).parents[1]
    loaded = [load_probe(path.parent) for path in sorted((root / "probes").glob("*/*/probe.yaml"))]
    rubrics = {probe.manifest.id: probe.manifest.judge_rubric for probe in loaded}
    labels = load_validation_labels(root / "data" / "judge-validation-labels.jsonl")
    assert len(labels) >= 50
    assert {label.human_labeler for label in labels} == {"Aziz Efe Taban"}
    capture_path = root / "data" / "judge-validation.jsonl"
    assert capture_path.is_file(), "fresh isolated judge capture is required"
    cases = load_validation_corpus(capture_path)

    summary = evaluate_validation_corpus(cases, rubrics)
    require_validation_gate(summary)

    assert summary.total == len(labels)
    assert len(set(summary.judge_models)) == 2
    assert all(":" in model for model in summary.judge_models)
    assert {case.probe_id for case in cases} == set(rubrics)
    assert [case.model_dump(include=set(JudgeValidationLabel.model_fields)) for case in cases] == [
        label.model_dump() for label in labels
    ]
    for case in cases:
        rubric = rubrics[case.probe_id]
        assert case.excerpt not in rubric.positive_examples
        assert case.excerpt not in rubric.negative_examples
