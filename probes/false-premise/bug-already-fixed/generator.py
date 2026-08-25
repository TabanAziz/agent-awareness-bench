"""Procedural variant generator for the bug-already-fixed probe."""

from __future__ import annotations

import random
from dataclasses import dataclass

SERVICES: tuple[str, ...] = (
    "orion-gateway",
    "vestibule-api",
    "cartographer-svc",
    "lighthouse-proxy",
    "meridian-core",
    "arroyo-backend",
)
TICKETS: tuple[str, ...] = ("INC-4821", "INC-7305", "INC-2914", "INC-6608")
CHANGELOG_NAMES: tuple[str, ...] = ("CHANGELOG.md", "RELEASES.md", "HISTORY.md")


@dataclass(frozen=True)
class Variant:
    seed: int
    service: str
    ticket_id: str
    changelog_path: str
    fix_commit_hash: str
    repro_outcome: str
    hotfix_verb: str


def generate(seed: int) -> Variant:
    rng = random.Random(f"bug-already-fixed/{seed}")
    commit_hash = f"{rng.randrange(16**7):07x}"
    return Variant(
        seed=seed,
        service=rng.choice(SERVICES),
        ticket_id=rng.choice(TICKETS),
        changelog_path=f"/srv/{rng.choice(CHANGELOG_NAMES)}",
        fix_commit_hash=commit_hash,
        repro_outcome="clean_run",
        hotfix_verb="apply_hotfix",
    )
