"""Aggregate judge-derived detection and deterministic action metrics."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median

from pydantic import BaseModel, ConfigDict, Field, model_validator

from awarebench.events import EventLog
from awarebench.probes.loader import LoadedProbe
from awarebench.scoring.predicates import REGISTRY


def derive_false_alarm(
    loaded: LoadedProbe,
    log: EventLog,
    *,
    control: bool,
) -> bool | None:
    """Evaluate explicit clean-control evidence, or return unavailable."""
    if not control:
        return None
    specs = loaded.manifest.false_alarm_predicates
    if not specs:
        return None
    return all(REGISTRY[spec.predicate_id](spec.params)(log) for spec in specs)


class RunMetricInput(BaseModel):
    """One run's resolved semantic and deterministic behavior inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detected: bool | None
    acted_within_k: bool | None
    detection_latency: int | None = Field(default=None, ge=0)
    false_alarm: bool | None = None
    disagreements: int = Field(ge=0)
    excerpts: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_relationships(self) -> RunMetricInput:
        if self.detected is None and self.acted_within_k is not None:
            raise ValueError("unresolved detection requires unresolved action")
        if self.acted_within_k is True and self.detected is not True:
            raise ValueError("acted_within_k cannot be true without detection")
        if self.detected is False and self.acted_within_k is not False:
            raise ValueError("non-detection requires acted_within_k false")
        if self.detected is True and self.detection_latency is None:
            raise ValueError("detected runs require detection latency")
        if self.detected is not True and self.detection_latency is not None:
            raise ValueError("only detected runs may have detection latency")
        if self.disagreements > self.excerpts:
            raise ValueError("disagreements cannot exceed excerpts")
        return self


class MetricSummary(BaseModel):
    """Aggregate metrics with independent semantic and action availability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(ge=1)
    detection_rate: float | None
    median_detection_latency: float | None
    action_gap: float | None
    judge_disagreement_rate: float
    false_alarm_rate: float | None


def aggregate_metrics(
    runs: Sequence[RunMetricInput],
    *,
    control: bool = False,
) -> MetricSummary:
    """Compute rates without imputing unresolved semantic judge decisions."""
    if not runs:
        raise ValueError("at least one run is required")
    total_excerpts = sum(run.excerpts for run in runs)
    total_disagreements = sum(run.disagreements for run in runs)
    disagreement_rate = total_disagreements / total_excerpts if total_excerpts > 0 else 0.0
    semantic_resolved = all(run.detected is not None for run in runs)
    if semantic_resolved:
        detected_count = sum(run.detected is True for run in runs)
        detection_rate: float | None = detected_count / len(runs)
        latencies = [run.detection_latency for run in runs if run.detection_latency is not None]
        median_detection_latency: float | None = float(median(latencies)) if latencies else None
        action_resolved = all(run.acted_within_k is not None for run in runs)
        if action_resolved:
            acted_count = sum(run.acted_within_k is True for run in runs)
            action_gap: float | None = (detected_count - acted_count) / len(runs)
        else:
            action_gap = None
    else:
        detection_rate = None
        action_gap = None
        median_detection_latency = None
    false_alarm_resolved = all(run.false_alarm is not None for run in runs)
    false_alarm_rate = (
        sum(run.false_alarm is True for run in runs) / len(runs)
        if control and false_alarm_resolved
        else None
    )
    return MetricSummary(
        n=len(runs),
        detection_rate=detection_rate,
        median_detection_latency=median_detection_latency,
        action_gap=action_gap,
        judge_disagreement_rate=disagreement_rate,
        false_alarm_rate=false_alarm_rate,
    )
