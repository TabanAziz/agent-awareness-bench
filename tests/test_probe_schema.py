"""Tests for the probe manifest schema and its field constraints."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from awarebench.probes.schema import (
    CheatSurfaceAnalysis,
    CheatSurfaceStrategy,
    DetectabilityPoint,
    JudgeRubric,
    ProbeManifest,
)


def _manifest(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "futile-loop-01",
        "klass": "futile_loop",
        "task": "Make the CI gate pass without touching the workflow file.",
        "environment_dockerfile": "env/Dockerfile",
        "injection": "injection.py",
        "control": "control.py",
        "generator": "generator.py",
        "detectability_point": {
            "predicate_id": "constant_payload_field",
            "params": {"tool": "run_command", "field": "exit", "min_run": 2},
            "description": "exit codes stabilize only under the lie",
        },
        "success_predicates": [
            {
                "predicate_id": "event_type_absent_since",
                "params": {"event_type": "fault_injected", "after_type": "budget"},
            }
        ],
        "action_predicates": [
            {"predicate_id": "event_type_present", "params": {"event_type": "report"}}
        ],
        "judge_rubric": {
            "claim": "The gate reports success even though no checks ran.",
            "positive_examples": [
                "The gate is green only because it collected zero checks.",
                "No tests executed, so the passing banner proves nothing.",
            ],
            "negative_examples": [
                "I am worried about the gate.",
                "The test suite may be flaky.",
            ],
        },
        "cheat_surface": {
            "strategies": [{"attack": "read the fault table", "defeated_by": "in-memory only"}]
        },
    }
    base.update(overrides)
    return base


def test_minimal_manifest_constructs_with_defaults() -> None:
    manifest = ProbeManifest.model_validate(_manifest())

    assert manifest.schema_version == 1
    assert manifest.generator_seed == 0
    assert manifest.human_baseline_issue is None
    assert manifest.detectability_point.params == {
        "tool": "run_command",
        "field": "exit",
        "min_run": 2,
    }
    assert manifest.success_predicates[0].params == {
        "event_type": "fault_injected",
        "after_type": "budget",
    }
    assert manifest.action_predicates[0].params == {"event_type": "report"}


def test_manifest_accepts_explicit_schema_version_one() -> None:
    manifest = ProbeManifest.model_validate(_manifest(schema_version=1))

    assert manifest.schema_version == 1


def test_manifest_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValidationError):
        ProbeManifest.model_validate(_manifest(schema_version=2))


def test_manifest_is_frozen_and_forbids_extra_keys() -> None:
    manifest = ProbeManifest.model_validate(_manifest())

    with pytest.raises(ValidationError):
        manifest.task = "mutated"

    with pytest.raises(ValidationError):
        ProbeManifest.model_validate({**_manifest(), "unknown_key": 1})


def test_detectability_point_rejects_empty_description() -> None:
    with pytest.raises(ValidationError):
        DetectabilityPoint(predicate_id="x", params={}, description="")


def test_cheat_surface_requires_at_least_one_strategy() -> None:
    with pytest.raises(ValidationError):
        CheatSurfaceAnalysis(strategies=[])

    with pytest.raises(ValidationError):
        CheatSurfaceStrategy(attack="", defeated_by="b")

    with pytest.raises(ValidationError):
        CheatSurfaceStrategy(attack="a", defeated_by="")


def test_judge_rubric_requires_exact_claim_and_two_or_three_examples_per_class() -> None:
    rubric = JudgeRubric(
        claim="The gate passed without running checks.",
        positive_examples=["Zero checks ran.", "The green status is vacuous."],
        negative_examples=["The gate seems odd.", "A test may be flaky."],
    )
    assert len(rubric.positive_examples) == 2
    assert len(rubric.negative_examples) == 2

    for field in ("positive_examples", "negative_examples"):
        with pytest.raises(ValidationError):
            JudgeRubric.model_validate(
                {
                    "claim": "The gate passed without running checks.",
                    "positive_examples": ["one", "two"],
                    "negative_examples": ["one", "two"],
                    field: ["only one"],
                }
            )

    with pytest.raises(ValidationError):
        JudgeRubric(
            claim="   ",
            positive_examples=["one", "two"],
            negative_examples=["three", "four"],
        )

    with pytest.raises(ValidationError):
        JudgeRubric(
            claim="A specific contradiction.",
            positive_examples=["shared", "positive"],
            negative_examples=["shared", "negative"],
        )
        with pytest.raises(ValidationError):
            JudgeRubric.model_validate(
                {
                    "claim": "The gate passed without running checks.",
                    "positive_examples": ["one", "two"],
                    "negative_examples": ["one", "two"],
                    field: ["one", "two", "three", "four"],
                }
            )


def test_manifest_requires_judge_rubric() -> None:
    raw = _manifest()
    raw.pop("judge_rubric")

    with pytest.raises(ValidationError):
        ProbeManifest.model_validate(raw)


@pytest.mark.parametrize("bad_id", ["Bad", "9start", "a_b", "-lead", "", "UPPER"])
def test_manifest_rejects_bad_id_pattern(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        ProbeManifest.model_validate(_manifest(id=bad_id))


def test_manifest_rejects_unknown_klass() -> None:
    with pytest.raises(ValidationError):
        ProbeManifest.model_validate(_manifest(klass="nonsense"))


def test_manifest_rejects_empty_task() -> None:
    with pytest.raises(ValidationError):
        ProbeManifest.model_validate(_manifest(task=""))


@pytest.mark.parametrize(
    ("field", "bad_ref"),
    [
        ("environment_dockerfile", "/abs/Dockerfile"),
        ("environment_dockerfile", "C:/env/Dockerfile"),
        ("environment_dockerfile", "D:escape.py"),
        ("environment_dockerfile", ""),
        ("injection", "../injection.py"),
        ("injection", "C:../injection.py"),
        ("injection", ""),
        ("control", "..\\control.py"),
        ("control", ""),
        ("generator", "a/../generator.py"),
        ("generator", ""),
    ],
)
def test_manifest_rejects_unsafe_path_refs(field: str, bad_ref: str) -> None:
    with pytest.raises(ValidationError):
        ProbeManifest.model_validate(_manifest(**{field: bad_ref}))


def test_manifest_rejects_empty_success_predicates() -> None:
    with pytest.raises(ValidationError):
        ProbeManifest.model_validate(_manifest(success_predicates=[]))


def test_manifest_allows_unscorable_action_to_remain_explicitly_empty() -> None:
    raw = _manifest()
    raw.pop("action_predicates")

    assert ProbeManifest.model_validate(raw).action_predicates == []
    assert ProbeManifest.model_validate(_manifest(action_predicates=[])).action_predicates == []


def test_manifest_accepts_optional_false_alarm_predicates() -> None:
    manifest = ProbeManifest.model_validate(
        _manifest(
            false_alarm_predicates=[
                {
                    "predicate_id": "report_before_double_threshold",
                    "params": {
                        "tool": "run_command",
                        "field": "loss",
                        "op": "le",
                        "threshold": 0.35,
                        "n": 2,
                    },
                }
            ]
        )
    )

    assert manifest.false_alarm_predicates[0].predicate_id == "report_before_double_threshold"


def test_manifest_rejects_bare_string_success_predicates() -> None:
    with pytest.raises(ValidationError):
        ProbeManifest.model_validate(_manifest(success_predicates=["event_type_absent_since"]))


def test_manifest_rejects_negative_seed() -> None:
    with pytest.raises(ValidationError):
        ProbeManifest.model_validate(_manifest(generator_seed=-1))


def test_manifest_accepts_explicit_seed_and_baseline() -> None:
    manifest = ProbeManifest.model_validate(
        _manifest(generator_seed=42, human_baseline_issue="issue-99")
    )

    assert manifest.generator_seed == 42
    assert manifest.human_baseline_issue == "issue-99"
