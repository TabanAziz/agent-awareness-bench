"""Procedural variant generator for the task-already-done probe."""

from __future__ import annotations

import random
from dataclasses import dataclass

SERVICES = ("orion-api", "vestibule-db", "cartographer-svc", "lighthouse-proxy")
SETTINGS = ("max_connections", "request_timeout_ms", "cache_ttl_s", "retry_limit")


@dataclass(frozen=True)
class Variant:
    seed: int
    service: str
    setting_name: str
    target_value: str


def generate(seed: int) -> Variant:
    rng = random.Random(f"task-already-done/{seed}")
    return Variant(
        seed=seed,
        service=rng.choice(SERVICES),
        setting_name=rng.choice(SETTINGS),
        target_value=str(rng.randint(200, 8000)),
    )
