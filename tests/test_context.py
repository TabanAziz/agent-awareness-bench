"""Tests for the context window simulation with silent compaction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

import awarebench.harness as harness_package
from awarebench.events import EventLog, EventType
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.context import (
    ContextWindow,
    DropPolicy,
    Message,
    crude_token_count,
    drop_oldest,
    drop_oldest_half,
)


def _fixed_counter(costs: dict[str, int]) -> Callable[[str], int]:
    def count(content: str) -> int:
        return costs[content]

    return count


def _make_window(
    max_tokens: int,
    costs: dict[str, int] | None = None,
    policy: DropPolicy | None = None,
) -> tuple[ContextWindow, EventLog, VirtualClock, CycleCounter]:
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    counter = _fixed_counter(costs) if costs is not None else None
    window = ContextWindow(
        event_log=log,
        clock=clock,
        cycles=cycles,
        max_tokens=max_tokens,
        token_counter=counter,
        policy=policy,
    )
    return window, log, clock, cycles


def _compaction_events(log: EventLog) -> list[Any]:
    return [event for event in log if event.type == EventType.COMPACTION]


# --- clean adds ---------------------------------------------------------------


def test_under_limit_adds_never_compact() -> None:
    window, log, *_ = _make_window(max_tokens=100)

    first = window.add("user", "hello")
    second = window.add("assistant", "hi there")

    assert [first.seq, second.seq] == [0, 1]
    assert window.compaction_count == 0
    assert _compaction_events(log) == []
    assert window.used_tokens == crude_token_count("hello") + crude_token_count("hi there")
    assert [m.content for m in window.messages()] == ["hello", "hi there"]


def test_messages_returns_defensive_copy() -> None:
    window, *_ = _make_window(max_tokens=100)
    window.add("user", "hello")

    snapshot = window.messages()

    assert isinstance(snapshot, tuple)
    window.add("assistant", "second")
    assert snapshot == (Message(seq=0, role="user", content="hello"),)
    assert len(window.messages()) == 2


def test_add_rejects_empty_role_without_logging() -> None:
    window, log, *_ = _make_window(max_tokens=10)

    with pytest.raises(ValueError, match="role"):
        window.add("", "content")

    assert len(log) == 0
    assert window.messages() == ()


@pytest.mark.parametrize("bad_max", [0, -5, True, 2.5])
def test_constructor_rejects_bad_max_tokens(bad_max: Any) -> None:
    with pytest.raises(ValueError):
        ContextWindow(
            event_log=EventLog(),
            clock=VirtualClock(),
            cycles=CycleCounter(),
            max_tokens=bad_max,
        )


# --- default policy: silent drop of the earliest message ----------------------


def test_default_policy_silently_drops_earliest_at_threshold() -> None:
    costs = {"constraint": 4, "filler-a": 3, "filler-b": 3, "trigger": 2}
    window, log, clock, cycles = _make_window(max_tokens=10, costs=costs)

    window.add("system", "constraint")
    window.add("user", "filler-a")
    window.add("user", "filler-b")
    assert window.used_tokens == 10
    clock.advance_us(500)
    cycles.advance()
    trigger = window.add("user", "trigger")

    assert trigger.seq == 3
    assert [m.content for m in window.messages()] == ["filler-a", "filler-b", "trigger"]
    assert window.used_tokens == 8
    assert window.compaction_count == 1
    events = _compaction_events(log)
    assert len(events) == 1
    assert events[0].payload["dropped_seq"] == [0]
    assert events[0].payload["freed_tokens"] == 4
    assert events[0].cycle == 1
    assert events[0].t_us == 500


# --- constraint survival vs threshold death -----------------------------------


def test_constraint_survives_half_cut_then_dies_at_threshold() -> None:
    costs: dict[str, int] = {"constraint": 2}
    for index in range(8):
        costs[f"f{index}"] = 2
    counter = _fixed_counter(costs)
    window, log, *_ = _make_window(max_tokens=10, costs=costs, policy=drop_oldest_half(counter))

    seqs = {name: window.add("user", name).seq for name in ("f0", "f1")}
    seqs["constraint"] = window.add("system", "constraint").seq
    for name in ("f3", "f4"):
        seqs[name] = window.add("user", name).seq
    # Window full at 5x2 tokens; the next add forces a half cut of the front.
    seqs["f5"] = window.add("user", "f5").seq

    contents = [m.content for m in window.messages()]
    assert contents == ["constraint", "f3", "f4", "f5"]
    compactions = _compaction_events(log)
    assert len(compactions) == 1
    assert compactions[0].payload["dropped_seq"] == [seqs["f0"], seqs["f1"]]
    assert compactions[0].payload["freed_tokens"] == 4

    # Refill; the next overflow puts the constraint at the front, where the
    # half cut finally takes it.
    window.add("user", "f6")
    window.add("user", "f7")

    contents = [m.content for m in window.messages()]
    assert contents == ["f4", "f5", "f6", "f7"]
    assert window.compaction_count == 2
    compactions = _compaction_events(log)
    assert len(compactions) == 2
    assert compactions[1].payload["dropped_seq"] == [seqs["constraint"], seqs["f3"]]


# --- oversized incoming message ------------------------------------------------


def test_incoming_larger_than_window_raises_without_logging() -> None:
    costs = {"tiny": 1, "huge": 11}
    window, log, *_ = _make_window(max_tokens=10, costs=costs)
    window.add("user", "tiny")

    with pytest.raises(ValueError, match="window holds"):
        window.add("user", "huge")

    assert len(log) == 0
    assert [m.content for m in window.messages()] == ["tiny"]
    assert window.used_tokens == 1


def test_do_nothing_policy_raises_without_corrupting_state() -> None:
    def noop(
        messages: Sequence[Message], tokens_to_free: int, incoming_tokens: int
    ) -> list[Message]:
        return list(messages)

    costs = {"a": 6, "b": 6}
    window, log, *_ = _make_window(max_tokens=10, costs=costs, policy=noop)
    window.add("user", "a")

    with pytest.raises(ValueError, match="freed no messages"):
        window.add("user", "b")

    assert [m.content for m in window.messages()] == ["a"]
    assert window.used_tokens == 6
    assert window.compaction_count == 0
    assert len(log) == 0


# --- built-in policies as units -------------------------------------------------


def _four_messages() -> list[Message]:
    contents = ("a", "b", "c", "d")
    return [Message(seq=index, role="user", content=c) for index, c in enumerate(contents)]


def test_drop_oldest_pops_front_until_enough_freed() -> None:
    counter = _fixed_counter({"a": 4, "b": 2, "c": 2, "d": 1})

    kept = drop_oldest(counter)(_four_messages(), 5, 2)

    assert [m.seq for m in kept] == [2, 3]


def test_drop_oldest_half_cuts_half_then_falls_back_to_front_popping() -> None:
    counter = _fixed_counter({"a": 4, "b": 2, "c": 2, "d": 1})
    messages = _four_messages()

    # Half cut frees a(4)+b(2)=6 >= 5: exactly the oldest half goes.
    kept = drop_oldest_half(counter)(messages, 5, 2)
    assert [m.seq for m in kept] == [2, 3]

    # Needing 9 exceeds the half cut (6): c(2) then d(1) are popped too.
    kept = drop_oldest_half(counter)(messages, 9, 2)
    assert kept == []


def test_drop_oldest_half_on_odd_count_cuts_floor_half() -> None:
    counter = _fixed_counter({"a": 2, "b": 2, "c": 2})
    messages = [
        Message(seq=index, role="user", content=c) for index, c in enumerate(("a", "b", "c"))
    ]

    kept = drop_oldest_half(counter)(messages, 2, 2)

    assert [m.seq for m in kept] == [1, 2]


# --- accounting ------------------------------------------------------------------


def test_used_tokens_and_compaction_count_track_state_exactly() -> None:
    costs = {"m0": 3, "m1": 3, "m2": 3, "m3": 3}
    window, log, *_ = _make_window(max_tokens=9, costs=costs)

    for name in ("m0", "m1", "m2"):
        window.add("user", name)
    assert window.used_tokens == 9

    window.add("user", "m3")

    expected = sum(costs[m.content] for m in window.messages())
    assert window.used_tokens == expected
    assert window.compaction_count == 1
    assert len(_compaction_events(log)) == 1


# --- determinism -----------------------------------------------------------------


def test_identical_windows_produce_identical_states_and_jsonl(tmp_path: Path) -> None:
    def build() -> tuple[ContextWindow, EventLog]:
        log = EventLog()
        clock = VirtualClock()
        cycles = CycleCounter()
        window = ContextWindow(event_log=log, clock=clock, cycles=cycles, max_tokens=10)
        return window, log

    script = [
        ("system", "you must obey the constraint"),
        ("user", "filler one"),
        ("user", "filler two"),
        ("user", "trigger three"),
        ("user", "trigger four"),
    ]

    window_a, log_a = build()
    window_b, log_b = build()
    for role, content in script:
        window_a.add(role, content)
        window_b.add(role, content)

    assert window_a.messages() == window_b.messages()
    assert window_a.used_tokens == window_b.used_tokens
    assert window_a.compaction_count == window_b.compaction_count == 1
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    log_a.write_jsonl(path_a)
    log_b.write_jsonl(path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


# --- package export surface -------------------------------------------------------


def test_harness_package_reexports_context_module() -> None:
    assert harness_package.ContextWindow is ContextWindow
    assert harness_package.Message is Message
    assert harness_package.DropPolicy is DropPolicy
    assert harness_package.drop_oldest is drop_oldest
    assert harness_package.drop_oldest_half is drop_oldest_half
    assert harness_package.crude_token_count is crude_token_count
