"""Pydantic schema for probe manifests.

Every model is frozen and forbids unknown keys, so a manifest is immutable
once loaded and typos in manifest keys are rejected instead of ignored.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from awarebench.scoring.predicates import PredicateParams

ProbeClass = Literal[
    "green_lying",
    "silent_truncation",
    "own_operation",
    "drifting_environment",
    "false_premise",
    "repo_conventions",
    "escape_hatch",
    "futile_loop",
    "invented_substrate",
    "abstract",
]

PATH_FIELDS: Final[tuple[str, ...]] = (
    "environment_dockerfile",
    "injection",
    "control",
    "generator",
)


def validate_relative_path(value: str) -> str:
    """Reject empty, backslashed, absolute, or parent-traversing manifest paths."""
    if not value:
        raise ValueError("path must be non-empty")
    if "\\" in value:
        raise ValueError(f"unsafe path {value!r}: forward slashes only")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"unsafe path {value!r}: must be relative")
    if ".." in PurePosixPath(value).parts:
        raise ValueError(f"unsafe path {value!r}: must not traverse parents")
    return value


class DetectabilityPoint(BaseModel):
    """The one observation that makes the probe's lie detectable in principle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    predicate_id: str
    params: PredicateParams = {}
    description: str = Field(min_length=1)


class CheatSurfaceStrategy(BaseModel):
    """One way an agent could cheat the probe, and what defeats it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attack: str = Field(min_length=1)
    defeated_by: str = Field(min_length=1)


class CheatSurfaceAnalysis(BaseModel):
    """Non-empty enumeration of the probe's cheat surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategies: list[CheatSurfaceStrategy] = Field(min_length=1)


class ProbeManifest(BaseModel):
    """Full probe manifest; the loader gates every field against reality."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    klass: ProbeClass
    task: str = Field(min_length=1)
    environment_dockerfile: str
    injection: str
    control: str
    generator: str
    detectability_point: DetectabilityPoint
    success_predicates: list[str] = Field(min_length=1)
    cheat_surface: CheatSurfaceAnalysis
    generator_seed: int = Field(default=0, ge=0)
    human_baseline_issue: str | None = None

    @field_validator(*PATH_FIELDS)
    @classmethod
    def _paths_must_be_safe(cls, value: str) -> str:
        """Apply the shared relative-path contract to every file reference."""
        return validate_relative_path(value)
