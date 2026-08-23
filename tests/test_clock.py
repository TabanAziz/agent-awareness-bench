"""Tests for the virtual clock and cycle counter."""

from __future__ import annotations

import pytest

from awarebench.harness.clock import CycleCounter, VirtualClock


def test_clock_starts_at_zero() -> None:
    assert VirtualClock().now_us == 0


def test_advance_accumulates_deterministically() -> None:
    clock = VirtualClock()
    clock.advance_us(250)
    clock.advance_us(750)
    clock.advance_us(0)

    assert clock.now_us == 1000


def test_clock_never_decreases() -> None:
    clock = VirtualClock()
    before = clock.now_us
    clock.advance_us(42)

    assert clock.now_us > before


def test_negative_delta_raises_and_leaves_clock_unchanged() -> None:
    clock = VirtualClock()
    clock.advance_us(10)

    with pytest.raises(ValueError, match="delta must be >= 0"):
        clock.advance_us(-1)

    assert clock.now_us == 10


def test_counter_starts_at_zero_and_advances_by_one() -> None:
    counter = CycleCounter()

    assert counter.current == 0
    assert counter.advance() == 1
    assert counter.advance() == 2
    assert counter.current == 2


def test_fresh_counters_produce_identical_sequences() -> None:
    first = CycleCounter()
    second = CycleCounter()
    sequence_first = [first.advance() for _ in range(5)]
    sequence_second = [second.advance() for _ in range(5)]

    assert sequence_first == sequence_second == [1, 2, 3, 4, 5]
