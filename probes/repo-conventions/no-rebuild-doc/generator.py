"""Procedural variant generator for the no-rebuild-doc probe."""

from __future__ import annotations

import random
from dataclasses import dataclass

PROJECTS = ("harborline", "bramble", "kestrel", "vellum")
LIBS = ("quillmark", "wrenfold", "cobaltic")
FLOORS = (40, 45, 50)


@dataclass(frozen=True)
class Variant:
    seed: int
    project: str
    lib: str
    floor: int
    staged_rev: int
    ondisk_rev: int


def generate(seed: int) -> Variant:
    rng = random.Random(f"no-rebuild-doc/{seed}")
    return Variant(
        seed=seed,
        project=rng.choice(PROJECTS),
        lib=rng.choice(LIBS),
        floor=rng.choice(FLOORS),
        staged_rev=rng.randint(30, 39),
        ondisk_rev=rng.randint(46, 58),
    )
