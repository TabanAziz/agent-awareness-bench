from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    seed: int


def generate(seed: int) -> Variant:
    return Variant(seed=seed)
