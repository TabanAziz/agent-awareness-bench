from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    seed: int
    freeze_poll: int


def generate(seed: int) -> Variant:
    rng = random.Random(f"frozen-dashboard/{seed}")
    return Variant(seed=seed, freeze_poll=rng.choice((2, 3, 4)))
