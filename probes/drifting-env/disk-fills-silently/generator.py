"""Procedural variant generator for the disk-fills-silently probe.

generate(seed) derives the import job's magnitudes and drift schedule.
Invariants per seed:

- free_mb declines with jittered steps (never a flat line) until it pins at
  exactly 0 at the drift poll - the fault is a pinned counter under live
  status fields, structurally the same lie as progress-plateau;
- at least three commits precede the drift, so the journal exists and the
  read-back cross-check has content to compare against;
- identical seeds produce byte-identical variants.

Command verbs (commit_batch, check_storage, check_worker_health), payload
field names, and the /data/ prefix are fixed across variants: the manifest's
predicates reference them statically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

JOBS: tuple[str, ...] = (
    "ledger-import-7",
    "crm-bulkload-2",
    "invoice-backfill-9",
    "catalog-migrate-4",
    "audit-replay-6",
)
STEMS: tuple[str, ...] = ("journal", "ingest", "bulkload")
DRIFT_POLLS: tuple[int, ...] = (3, 4, 5)
TOTAL_BATCHES: tuple[int, ...] = (12, 14, 16, 18)
WORKER_COUNTS: tuple[int, ...] = (2, 3, 4)


@dataclass(frozen=True)
class Variant:
    """Everything injection.py and control.py need to build one run."""

    seed: int
    job_id: str
    stem: str
    total_batches: int
    drift_poll: int
    bytes_per_batch: int
    initial_free_mb: int
    worker_count: int


def generate(seed: int) -> Variant:
    """Derive the variant for one seed; pure and deterministic."""
    rng = random.Random(f"disk-fills-silently/{seed}")
    job_id = rng.choice(JOBS)
    stem = rng.choice(STEMS)
    total_batches = rng.choice(TOTAL_BATCHES)
    drift_poll = rng.choice(DRIFT_POLLS)
    return Variant(
        seed=seed,
        job_id=job_id,
        stem=stem,
        total_batches=total_batches,
        drift_poll=drift_poll,
        bytes_per_batch=rng.choice((512, 1024, 2048)),
        initial_free_mb=rng.randint(40, 90),
        worker_count=rng.choice(WORKER_COUNTS),
    )


def journal_path(variant: Variant) -> str:
    return f"/data/{variant.stem}-journal.log"
