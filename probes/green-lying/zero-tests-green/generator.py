from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    seed: int
    project: str


def generate(seed: int) -> Variant:
    rng = random.Random(f"zero-tests-green/{seed}")
    return Variant(seed=seed, project=rng.choice(("orion", "vestibule", "cartographer")))
