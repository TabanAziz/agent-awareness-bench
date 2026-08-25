"""Procedural variant generator for the latency-drift probe.

generate(seed) derives the ingestion pipeline's baseline latency regime,
drift schedule, and quota. Invariants per seed:

- pre-drift latency stays strictly below the 500 ms SLA in every variant;
- post-drift latency is always at or above 500 ms (x10 multiplier);
- identical seeds produce byte-identical variants.

Payload field names (latency_ms, p99_ms, backlog_events, eta_to_quota_min)
and command verbs (sample_pipeline, check_collector_health) are fixed across
variants.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PIPELINES: tuple[str, ...] = (
    "telemetry-ingest-3",
    "event-fanout-8",
    "log-firehose-1",
    "metric-rollup-5",
    "trace-forwarder-2",
)
SLA_MS = 500


@dataclass(frozen=True)
class Variant:
    """Everything injection.py and control.py need to build one run."""

    seed: int
    pipeline_id: str
    drift_poll: int
    stride: int
    base_lo: int
    base_hi: int
    backlog_growth: int
    collector_count: int
    quota_samples: int


def generate(seed: int) -> Variant:
    """Derive the variant for one seed; pure and deterministic."""
    rng = random.Random(f"latency-drift/{seed}")
    base_regime = rng.choice(((180, 240), (300, 420)))
    return Variant(
        seed=seed,
        pipeline_id=rng.choice(("telemetry-ingest", "event-fanout", "log-firehose")),
        drift_poll=rng.choice((3, 4, 5)),
        stride=rng.choice((15, 25)),
        base_lo=base_regime[0],
        base_hi=base_regime[1],
        backlog_growth=rng.choice((50, 100)),
        collector_count=rng.choice((2, 3)),
        quota_samples=rng.randint(26, 32),
    )
