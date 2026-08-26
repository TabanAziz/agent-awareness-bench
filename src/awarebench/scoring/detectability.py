"""Locate the exact event prefix where a probe becomes detectable."""

from __future__ import annotations

from dataclasses import dataclass

from awarebench.events import EventLog
from awarebench.probes.loader import LoadedProbe
from awarebench.scoring.predicates import REGISTRY


@dataclass(frozen=True)
class DetectabilityBoundary:
    """The first event whose inclusive prefix satisfies the manifest predicate."""

    seq: int
    cycle: int


def detectability_boundary(
    loaded: LoadedProbe,
    log: EventLog,
) -> DetectabilityBoundary | None:
    """Return the first satisfying event, preserving within-cycle ordering."""
    spec = loaded.manifest.detectability_point
    predicate = REGISTRY[spec.predicate_id](spec.params)
    prefix = EventLog()
    for event in log:
        prefix.append(event.type, event.cycle, event.t_us, event.payload)
        if predicate(prefix):
            return DetectabilityBoundary(seq=event.seq, cycle=event.cycle)
    return None
