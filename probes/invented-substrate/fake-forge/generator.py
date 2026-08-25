from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    seed: int
    drift_poll: int


def generate(seed: int) -> Variant:
    rng = random.Random(f"fake-forge/{seed}")
    return Variant(seed=seed, drift_poll=rng.randint(2, 4))
