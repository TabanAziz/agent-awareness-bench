"""Post-run scoring: instantiate a probe's predicates over its event log.

The loader guarantees every referenced predicate is factory-able; this module
instantiates each spec against a finished log and combines the results. The
machine layer is the deterministic floor — semantic grading of report text is
a separate judge concern and never happens here.
"""

from __future__ import annotations

from awarebench.events import EventLog
from awarebench.probes.loader import LoadedProbe
from awarebench.probes.schema import PredicateSpec
from awarebench.scoring.predicates import REGISTRY, PredicateFactory


def _specs_for(loaded: LoadedProbe, *, control: bool) -> list[PredicateSpec]:
    """Success predicates by default; control runs use their own set when given."""
    manifest = loaded.manifest
    if control and manifest.control_predicates:
        return manifest.control_predicates
    return manifest.success_predicates


def evaluate(
    loaded: LoadedProbe,
    log: EventLog,
    registry: dict[str, PredicateFactory] | None = None,
    *,
    control: bool = False,
) -> dict[str, bool]:
    """Run every predicate spec for the run kind against the log.

    Returns a mapping keyed by predicate id. Duplicate ids collapse to the
    last result; a repeated id with identical params is redundant anyway.
    """
    reg = registry if registry is not None else REGISTRY
    results: dict[str, bool] = {}
    for spec in _specs_for(loaded, control=control):
        results[spec.predicate_id] = reg[spec.predicate_id](spec.params)(log)
    return results


def passed(results: dict[str, bool]) -> bool:
    """A run passes when every evaluated predicate holds."""
    return all(results.values())
