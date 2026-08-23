"""Deterministic virtual clock and cycle counter."""

from __future__ import annotations


class VirtualClock:
    """Deterministic virtual clock in integer microseconds; starts at 0.

    Reads no wall-clock time; time only moves via advance_us.
    """

    def __init__(self) -> None:
        self._now_us: int = 0

    @property
    def now_us(self) -> int:
        """Current virtual time in microseconds."""
        return self._now_us

    def advance_us(self, delta: int) -> None:
        """Advance the clock by delta microseconds; delta must be >= 0."""
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        self._now_us += delta


class CycleCounter:
    """Deterministic agent-step counter; starts at 0."""

    def __init__(self) -> None:
        self._current: int = 0

    @property
    def current(self) -> int:
        """Current cycle number."""
        return self._current

    def advance(self) -> int:
        """Move to the next cycle and return its number."""
        self._current += 1
        return self._current
