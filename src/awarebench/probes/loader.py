"""Strict probe loader: the first automated legitimacy gate.

Every rejection is a ProbeGateError whose message names the failed gate, so a
probe author sees exactly which legitimacy requirement their probe misses:
manifest presence, path safety, detectability point, machine-checkable
predicates, control variant, procedural generator, injection module,
environment dockerfile, and cheat surface analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from awarebench.probes.schema import PATH_FIELDS, ProbeManifest, validate_relative_path
from awarebench.scoring.action import ACTION_REGISTRY
from awarebench.scoring.predicates import REGISTRY, PredicateFactory
from awarebench.solvability import SOLVABILITY_FILENAME, load_solvability

_FILE_GATES: Final[dict[str, str]] = {
    "environment_dockerfile": "missing environment dockerfile",
    "injection": "missing injection module",
    "control": "missing control variant",
    "generator": "missing procedural generator",
}


class ProbeGateError(Exception):
    """A probe failed a legitimacy gate; the message names the failed gate."""


class LoadedProbe(BaseModel):
    """Validated probe manifest plus fully resolved artifact paths.

    Direct construction bypasses every gate; load_probe is the gated contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: ProbeManifest
    probe_dir: Path
    environment_dockerfile: Path
    injection: Path
    control: Path
    generator: Path


def load_probe(
    probe_dir: Path,
    registry: dict[str, PredicateFactory] | None = None,
    *,
    require_solvability: bool = False,
) -> LoadedProbe:
    """Load and gate-check the probe under probe_dir.

    Gate order: manifest present and parseable (a), raw path safety (h),
    raw presence of a detectability point (c) and cheat strategies (g),
    schema validation (b), machine-checkability against the registry
    (d, e), then artifact files on disk (f). The raw checks run before
    schema validation so each gate's name surfaces even when the schema
    would reject the same defect. Every failure raises ProbeGateError.
    """
    active_registry = REGISTRY if registry is None else registry
    active_action_registry = ACTION_REGISTRY if registry is None else active_registry
    root = Path(probe_dir)

    raw = _load_raw_manifest(root)
    _ensure_raw_paths_safe(raw)
    _ensure_raw_detectability_present(raw)
    _ensure_raw_cheat_surface_present(raw)
    try:
        manifest = ProbeManifest.model_validate(raw)
    except ValidationError as exc:
        msg = f"invalid manifest: {exc}"
        raise ProbeGateError(msg) from exc

    point = manifest.detectability_point
    factory = active_registry.get(point.predicate_id)
    if factory is None:
        msg = (
            f"unknown detectability predicate '{point.predicate_id}': "
            "detectability point not machine-checkable"
        )
        raise ProbeGateError(msg)
    try:
        factory(point.params)
    except ValueError as exc:
        msg = f"detectability point not machine-checkable: {exc}"
        raise ProbeGateError(msg) from exc

    for spec in manifest.success_predicates:
        factory = active_registry.get(spec.predicate_id)
        if factory is None:
            msg = f"unknown success predicate '{spec.predicate_id}': not machine-checkable"
            raise ProbeGateError(msg)
        try:
            factory(spec.params)
        except ValueError as exc:
            msg = f"success predicate '{spec.predicate_id}' not machine-checkable: {exc}"
            raise ProbeGateError(msg) from exc

    for spec in manifest.action_predicates:
        action_factory = active_action_registry.get(spec.predicate_id)
        if action_factory is None:
            msg = f"unknown action predicate '{spec.predicate_id}': not machine-checkable"
            raise ProbeGateError(msg)
        try:
            action_factory(spec.params)
        except ValueError as exc:
            msg = f"action predicate '{spec.predicate_id}' not machine-checkable: {exc}"
            raise ProbeGateError(msg) from exc

    for spec in manifest.false_alarm_predicates:
        factory = active_registry.get(spec.predicate_id)
        if factory is None:
            msg = f"unknown false-alarm predicate '{spec.predicate_id}': not machine-checkable"
            raise ProbeGateError(msg)
        try:
            factory(spec.params)
        except ValueError as exc:
            msg = f"false-alarm predicate '{spec.predicate_id}' not machine-checkable: {exc}"
            raise ProbeGateError(msg) from exc

    for spec in manifest.control_predicates:
        factory = active_registry.get(spec.predicate_id)
        if factory is None:
            msg = f"unknown control predicate '{spec.predicate_id}': not machine-checkable"
            raise ProbeGateError(msg)
        try:
            factory(spec.params)
        except ValueError as exc:
            msg = f"control predicate '{spec.predicate_id}' not machine-checkable: {exc}"
            raise ProbeGateError(msg) from exc

    if require_solvability:
        _ensure_solvability_passes(root, manifest.id)

    resolved: dict[str, Path] = {}
    for field_name in PATH_FIELDS:
        ref = getattr(manifest, field_name)
        target = root / ref
        if not target.is_file():
            msg = f"{_FILE_GATES[field_name]}: expected file {ref!r} in the probe directory"
            raise ProbeGateError(msg)
        resolved[field_name] = target.resolve()

    return LoadedProbe(
        manifest=manifest,
        probe_dir=root,
        environment_dockerfile=resolved["environment_dockerfile"],
        injection=resolved["injection"],
        control=resolved["control"],
        generator=resolved["generator"],
    )


def _ensure_solvability_passes(root: Path, probe_id: str) -> None:
    """Reject probes lacking ten cold runs or the documented eight-run threshold."""
    path = root / SOLVABILITY_FILENAME
    if not path.is_file():
        raise ProbeGateError(f"missing solvability record: expected {SOLVABILITY_FILENAME}")
    try:
        result = load_solvability(path)
    except ValueError as exc:
        raise ProbeGateError(str(exc)) from exc
    if result.probe_id != probe_id:
        raise ProbeGateError(
            f"solvability probe id mismatch: expected {probe_id!r}, got {result.probe_id!r}"
        )
    if result.passed_count < result.threshold:
        raise ProbeGateError(
            f"solvability below threshold: {result.passed_count}/{result.count} < {result.threshold}"
        )


def _load_raw_manifest(root: Path) -> dict[str, Any]:
    """Read and parse probe.yaml; any failure raises the manifest gate."""
    manifest_path = root / "probe.yaml"
    if not manifest_path.is_file():
        msg = "missing manifest: no probe.yaml in the probe directory"
        raise ProbeGateError(msg)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        msg = f"missing manifest: probe.yaml unreadable or unparseable: {exc}"
        raise ProbeGateError(msg) from exc
    if not isinstance(raw, dict):
        msg = "missing manifest: probe.yaml must parse to a mapping"
        raise ProbeGateError(msg)
    return raw


def _ensure_raw_paths_safe(raw: dict[str, Any]) -> None:
    """Apply the relative-path contract to raw string refs before schema runs."""
    for field_name in PATH_FIELDS:
        value = raw.get(field_name)
        if not isinstance(value, str):
            continue  # non-str refs are schema-level failures, not path-safety ones
        try:
            validate_relative_path(value)
        except ValueError as exc:
            msg = f"unsafe path in manifest: field '{field_name}': {exc}"
            raise ProbeGateError(msg) from exc


def _ensure_raw_detectability_present(raw: dict[str, Any]) -> None:
    """Reject manifests with no detectability point before schema normalizes them."""
    point = raw.get("detectability_point")
    if not isinstance(point, dict) or not point:
        msg = "missing detectability point"
        raise ProbeGateError(msg)


def _ensure_raw_cheat_surface_present(raw: dict[str, Any]) -> None:
    """Reject manifests whose cheat surface analysis carries no strategies."""
    surface = raw.get("cheat_surface")
    strategies = surface.get("strategies") if isinstance(surface, dict) else None
    if not isinstance(strategies, list) or not strategies:
        msg = "missing cheat surface analysis"
        raise ProbeGateError(msg)
