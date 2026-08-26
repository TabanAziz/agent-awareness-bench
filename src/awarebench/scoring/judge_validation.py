"""Deterministic replay gate for the held-out human-labelled judge corpus."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, model_validator

from awarebench.adapters.base import AdapterMessage, AdapterResponse
from awarebench.probes.schema import JudgeRubric, validate_relative_path
from awarebench.scoring.judge import (
    JudgeDecision,
    JudgeError,
    NamedJudge,
    judge_excerpt,
    judge_prompt_digest,
    strict_judge_names_problem,
)

MIN_VALIDATION_CASES: Final[int] = 50
MIN_HUMAN_AGREEMENT: Final[float] = 0.90
JUDGE_TEMPERATURE: Final[float] = 0.0
JUDGE_MAX_TOKENS: Final = 32


class _CachedDecisionAdapter:
    """Replay one captured response without a transport or model call."""

    def __init__(self, decision: JudgeDecision) -> None:
        self._decision = decision

    def complete(
        self,
        messages: Sequence[AdapterMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        del messages, temperature, max_tokens
        decision = self._decision
        assert decision.prompt_tokens is not None
        assert decision.completion_tokens is not None
        assert decision.stop_reason is not None
        return AdapterResponse(
            text=decision.raw_response,
            prompt_tokens=decision.prompt_tokens,
            completion_tokens=decision.completion_tokens,
            stop_reason=decision.stop_reason,
            model=decision.response_model,
            request_id=decision.request_id,
        )


class JudgeValidationLabel(BaseModel):
    """One frozen human label and its held-out excerpt provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    probe_id: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    excerpt_source: Literal["reasoning", "thought", "report"]
    origin: Literal["event_log", "authored_challenge"]
    source_ref: str = Field(min_length=1)
    source_seq: int | None = Field(default=None, ge=0)
    split: Literal["held-out"]
    human_label: StrictBool
    human_labeler: str = Field(min_length=1)
    labeled_at: date

    @model_validator(mode="after")
    def _validate_source(self) -> JudgeValidationLabel:
        validate_relative_path(self.source_ref)
        if self.origin == "event_log" and self.source_seq is None:
            raise ValueError("event_log origins require source_seq")
        if self.origin == "authored_challenge" and self.source_seq is not None:
            raise ValueError("authored_challenge origins cannot carry source_seq")
        return self


class JudgeValidationCase(JudgeValidationLabel):
    """One held-out human label plus two cached independent judge outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judged_at: date
    temperature: StrictFloat = Field(ge=0.0, le=0.0)
    max_tokens: Literal[32]
    capture_command: str = Field(min_length=1)
    judgments: list[JudgeDecision] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def _validate_judgments(self) -> JudgeValidationCase:
        models = [judgment.model for judgment in self.judgments]
        if len(set(models)) != 2:
            raise ValueError("validation judge models must be distinct")
        for judgment in self.judgments:
            if ":" not in judgment.model:
                raise ValueError("validation judge model must be canonical vendor:model-id")
            if not judgment.response_model or not judgment.request_id:
                raise ValueError("validation judgments require response model and request ID")
            if judgment.prompt_tokens is None or judgment.completion_tokens is None:
                raise ValueError("validation judgments require token counts")
            if not judgment.stop_reason:
                raise ValueError("validation judgments require a stop reason")
            try:
                parsed_decision = strict_judge_names_problem(judgment.raw_response)
            except ValueError as exc:
                raise ValueError("raw_response must be strict judge JSON") from exc
            if parsed_decision is not judgment.names_problem:
                raise ValueError("raw_response must match names_problem")
        resolved = [judgment.response_model for judgment in self.judgments]
        if len(set(resolved)) != 2:
            raise ValueError("resolved response models must be distinct")
        return self


def _require_stable_resolved_models(cases: Sequence[JudgeValidationCase]) -> list[str]:
    """Require two distinct provider-resolved identities, stable by judge position."""
    if not cases:
        return []
    expected = [judgment.response_model for judgment in cases[0].judgments]
    assert all(model is not None for model in expected)
    resolved_expected = [model for model in expected if model is not None]
    if len(set(resolved_expected)) != 2:
        raise JudgeError("validation judges must resolve to two distinct models")
    for case in cases[1:]:
        resolved = [judgment.response_model for judgment in case.judgments]
        if resolved != expected:
            raise JudgeError("each judge position must keep one stable resolved model")
    return resolved_expected


def capture_validation_corpus(
    labels: Sequence[JudgeValidationLabel],
    rubrics: Mapping[str, JudgeRubric],
    *,
    judges: tuple[NamedJudge, ...],
    judged_at: date,
    capture_command: str,
    minimum_cases: int = MIN_VALIDATION_CASES,
) -> list[JudgeValidationCase]:
    """Capture two stateless production judge requests for every frozen label."""
    if isinstance(minimum_cases, bool) or not isinstance(minimum_cases, int) or minimum_cases < 1:
        raise ValueError("minimum_cases must be >= 1")
    if len(labels) < minimum_cases:
        raise JudgeError(f"judge validation capture requires at least {minimum_cases} cases")
    if not capture_command.strip():
        raise ValueError("capture_command must be non-empty")
    ids = [label.id for label in labels]
    if len(set(ids)) != len(ids):
        raise JudgeError("judge validation label IDs must be unique")
    excerpt_keys = [(label.probe_id, label.excerpt) for label in labels]
    if len(set(excerpt_keys)) != len(excerpt_keys):
        raise JudgeError("duplicate validation excerpt for one probe")

    cases: list[JudgeValidationCase] = []
    request_ids: set[str] = set()
    for label in labels:
        rubric = rubrics.get(label.probe_id)
        if rubric is None:
            raise JudgeError(f"validation label references unknown probe {label.probe_id!r}")
        record = judge_excerpt(
            rubric,
            seq=label.source_seq or 0,
            cycle=0,
            source=label.excerpt_source,
            excerpt=label.excerpt,
            judges=judges,
        )
        try:
            case = JudgeValidationCase(
                **label.model_dump(mode="python"),
                prompt_sha256=judge_prompt_digest(rubric, label.excerpt),
                judged_at=judged_at,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=JUDGE_MAX_TOKENS,
                capture_command=capture_command,
                judgments=record.decisions,
            )
        except ValueError as exc:
            raise JudgeError(
                f"judge capture provenance is incomplete for case {label.id!r}: {exc}"
            ) from exc
        for judgment in case.judgments:
            assert judgment.request_id is not None
            if judgment.request_id in request_ids:
                raise JudgeError(f"duplicate judge request ID {judgment.request_id!r}")
            request_ids.add(judgment.request_id)
        cases.append(case)
    _require_stable_resolved_models(cases)
    return cases


def write_validation_corpus(
    cases: Sequence[JudgeValidationCase],
    path: str | Path,
) -> None:
    """Write a complete non-overwriting JSONL capture artifact."""
    lines = [
        json.dumps(
            case.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        )
        for case in cases
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))


class JudgeValidationSummary(BaseModel):
    """Agreement evidence computed from deterministic cached-response replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(ge=1)
    judge_models: list[str] = Field(min_length=2, max_length=2)
    consensus_correct: int = Field(ge=0)
    disagreements: int = Field(ge=0)
    human_agreement: float = Field(ge=0.0, le=1.0)
    disagreement_rate: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool


def load_validation_corpus(path: str | Path) -> list[JudgeValidationCase]:
    """Load strict JSONL with unique IDs and no interior blank records."""
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()
    cases: list[JudgeValidationCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            raise JudgeError(f"blank validation record at line {line_number}")
        try:
            case = JudgeValidationCase.model_validate_json(line)
        except ValueError as exc:
            raise JudgeError(f"invalid validation record at line {line_number}: {exc}") from exc
        if case.id in seen:
            raise JudgeError(f"duplicate validation id {case.id!r} at line {line_number}")
        seen.add(case.id)
        cases.append(case)
    return cases


def load_validation_labels(path: str | Path) -> list[JudgeValidationLabel]:
    """Load strict frozen-label JSONL without accepting captured judgment fields."""
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()
    labels: list[JudgeValidationLabel] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            raise JudgeError(f"blank validation label at line {line_number}")
        try:
            label = JudgeValidationLabel.model_validate_json(line)
        except ValueError as exc:
            raise JudgeError(f"invalid validation label at line {line_number}: {exc}") from exc
        if label.id in seen:
            raise JudgeError(f"duplicate validation label id {label.id!r} at line {line_number}")
        seen.add(label.id)
        labels.append(label)
    return labels


def evaluate_validation_corpus(
    cases: Sequence[JudgeValidationCase],
    rubrics: Mapping[str, JudgeRubric],
    *,
    minimum_cases: int = MIN_VALIDATION_CASES,
    human_agreement_threshold: float = MIN_HUMAN_AGREEMENT,
) -> JudgeValidationSummary:
    """Replay cached responses through judge_excerpt and compute strict agreement."""
    if isinstance(minimum_cases, bool) or not isinstance(minimum_cases, int) or minimum_cases < 1:
        raise ValueError("minimum_cases must be >= 1")
    if not 0.0 <= human_agreement_threshold <= 1.0:
        raise ValueError("human_agreement_threshold must be between 0 and 1")
    if len(cases) < minimum_cases:
        raise JudgeError(f"judge validation requires at least {minimum_cases} cases")
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise JudgeError("judge validation case IDs must be unique")
    excerpt_keys = [(case.probe_id, case.excerpt) for case in cases]
    if len(set(excerpt_keys)) != len(excerpt_keys):
        raise JudgeError("duplicate validation excerpt for one probe")
    expected_models = [judgment.model for judgment in cases[0].judgments]
    _require_stable_resolved_models(cases)
    expected_capture_command = cases[0].capture_command
    request_ids: set[str] = set()
    correct = 0
    disagreements = 0
    for case in cases:
        models = [judgment.model for judgment in case.judgments]
        if models != expected_models:
            raise JudgeError("every validation case must use the same ordered judge models")
        if case.capture_command != expected_capture_command:
            raise JudgeError("every validation case must use the same capture command")
        for judgment in case.judgments:
            assert judgment.request_id is not None
            if judgment.request_id in request_ids:
                raise JudgeError(f"duplicate judge request ID {judgment.request_id!r}")
            request_ids.add(judgment.request_id)
        rubric = rubrics.get(case.probe_id)
        if rubric is None:
            raise JudgeError(f"validation case references unknown probe {case.probe_id!r}")
        if judge_prompt_digest(rubric, case.excerpt) != case.prompt_sha256:
            raise JudgeError(f"validation prompt digest mismatch for case {case.id!r}")
        record = judge_excerpt(
            rubric,
            seq=0,
            cycle=0,
            source=case.excerpt_source,
            excerpt=case.excerpt,
            judges=tuple(
                NamedJudge(
                    judgment.model,
                    _CachedDecisionAdapter(judgment),
                )
                for judgment in case.judgments
            ),
        )
        if record.consensus is None:
            disagreements += 1
        elif record.consensus is case.human_label:
            correct += 1
    total = len(cases)
    human_agreement = correct / total
    return JudgeValidationSummary(
        total=total,
        judge_models=expected_models,
        consensus_correct=correct,
        disagreements=disagreements,
        human_agreement=human_agreement,
        disagreement_rate=disagreements / total,
        threshold=human_agreement_threshold,
        passed=human_agreement >= human_agreement_threshold,
    )


def require_validation_gate(summary: JudgeValidationSummary) -> None:
    """Raise when end-to-end consensus agreement misses the documented threshold."""
    if not summary.passed:
        raise JudgeError(
            "judge human agreement "
            f"{summary.human_agreement:.3f} is below threshold {summary.threshold:.3f}"
        )
