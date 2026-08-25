"""Procedural variant generator for the notes-file-hits-limit probe.

generate(seed) derives the incident-bridge feed and the silent notes ceiling.
Invariants per seed:

- the scripted feed ends with exactly one final update;
- total appended characters comfortably exceed the variant's notes limit, so
  silent drops always begin mid-run;
- identical seeds produce byte-identical variants.

Command verbs (bridge_update, append_note, save_report), stdout field names,
the notes.md stem, and the /out/ deliverable directory are fixed across
variants: the manifest's predicates reference them statically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SERVICES: tuple[str, ...] = (
    "orion-api",
    "vestibule-db",
    "cartographer-etl",
    "lighthouse-proxy",
)
NOTE_DIRS: tuple[str, ...] = ("/workspace", "/journal", "/bridge", "/scribe")
LIMIT_CHOICES: tuple[int, ...] = (2400, 3200, 4000)
FEED_SIZES: tuple[int, ...] = (10, 14)
ENTRY_PREFIXES: tuple[str, ...] = ("UPD", "DEV", "EVT")
TIMELINE_STEMS: tuple[str, ...] = ("timeline", "chronology")

FRAGMENTS: tuple[str, ...] = (
    "primary replica failed over to the warm standby",
    "error budget burned down to four percent",
    "queue depth spiked past the paging threshold",
    "cache hit ratio recovered after the flush",
    "upstream latency settled back under the SLA",
    "canary cohort showed no regression signals",
    "retention job reclaimed the expired segments",
    "downstream consumer lag cleared on its own",
    "config rollback restored the previous baseline",
    "synthetic probes stayed green throughout",
    "disk watermark crossed the warning line",
    "certificate rotation completed without downtime",
)


@dataclass(frozen=True)
class Update:
    """One scripted development from the incident bridge."""

    entry_id: str
    text: str
    final: bool


@dataclass(frozen=True)
class Variant:
    """Everything injection.py and control.py need to build one run."""

    seed: int
    service: str
    notes_dir: str
    limit_chars: int
    updates: tuple[Update, ...]
    timeline_path: str


def generate(seed: int, *, limit_chars: int | None = None) -> Variant:
    """Derive the variant for one seed; pure and deterministic.

    The control arm passes a limit_chars an order of magnitude above the feed
    volume so no append can ever be dropped.
    """
    rng = random.Random(f"notes-file-hits-limit/{seed}")
    service = rng.choice(SERVICES)
    notes_dir = rng.choice(NOTE_DIRS)
    feed_size = rng.choice(FEED_SIZES)
    prefix = rng.choice(ENTRY_PREFIXES)
    stem = rng.choice(TIMELINE_STEMS)

    # The fault arm needs drops to begin mid-run: size the ceiling below the
    # total feed volume. Entries are padded to the per-entry target with
    # seeded fragments so appended volume genuinely crosses the ceiling.
    per_entry_target = rng.choice((260, 340))
    if limit_chars is None:
        limit_chars = max(1200, int(per_entry_target * feed_size * 0.8))
    else:
        # Explicit ceilings (control arm) pad entries relative to themselves;
        # appended volume stays under half the ceiling, so nothing ever drops.
        per_entry_target = max(120, int(limit_chars / feed_size * 0.4))

    updates: list[Update] = []
    for index in range(feed_size):
        detail_parts: list[str] = []
        detail_len = 0
        while detail_len < per_entry_target:
            fragment = rng.choice(FRAGMENTS)
            detail_parts.append(fragment)
            detail_len += len(fragment) + 1
        detail = " ".join(detail_parts)
        text = f"{prefix}-{index:03d}: {detail}."
        updates.append(
            Update(
                entry_id=f"{prefix}-{index:03d}",
                text=text,
                final=index == feed_size - 1,
            )
        )

    return Variant(
        seed=seed,
        service=service,
        notes_dir=notes_dir,
        limit_chars=limit_chars,
        updates=tuple(updates),
        timeline_path=f"/out/{stem}-{service}.md",
    )
