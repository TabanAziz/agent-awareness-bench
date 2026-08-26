"""Tests for the context window simulation with silent compaction."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from awarebench.adapters.base import message_token_text
from awarebench.events import EventLog, EventType, JsonValue
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


def test_transcript_returns_agent_safe_role_content_pairs() -> None:
    window, *_ = _make_window(max_tokens=100)

    window.add("system", "constraint")
    window.add("user", "question")

    assert window.transcript() == (("system", "constraint"), ("user", "question"))
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in window.transcript())


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


@pytest.mark.parametrize("bad_seq", [-1, True, 1.5])
def test_message_rejects_bad_seq_on_direct_construction(bad_seq: Any) -> None:
    with pytest.raises(ValueError):
        Message(seq=bad_seq, role="user", content="x")


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
    window, log, *_ = _make_window(max_tokens=10, costs=costs, policy=drop_oldest_half)

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


def test_replay_metadata_counts_toward_incoming_limit_without_side_effects() -> None:
    window, log, *_ = _make_window(max_tokens=8)
    metadata: dict[str, JsonValue] = {
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "x" * 64},
        ]
    }

    with pytest.raises(ValueError, match="window holds"):
        window.add("assistant", "ok", metadata=metadata)

    assert window.used_tokens == 0
    assert window.messages() == ()
    assert len(log) == 0


def test_compaction_freed_tokens_include_replay_metadata() -> None:
    metadata: dict[str, JsonValue] = {
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "ciphertext"},
        ]
    }
    first_text = message_token_text("ok", metadata)
    first_cost = crude_token_count(first_text)
    window, log, *_ = _make_window(max_tokens=first_cost)
    window.add("assistant", "ok", metadata=metadata)

    window.add("user", "next")

    assert window.used_tokens == crude_token_count("next")
    [event] = _compaction_events(log)
    assert event.payload["freed_tokens"] == first_cost


def test_nested_metadata_mutation_cannot_change_internal_accounting() -> None:
    metadata: dict[str, JsonValue] = {
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "ciphertext"},
        ]
    }
    expected_wire: dict[str, JsonValue] = copy.deepcopy(
        {
            "role": "assistant",
            "content": "ok",
            **metadata,
        }
    )
    first_cost = crude_token_count(message_token_text("ok", metadata))
    window, log, *_ = _make_window(max_tokens=first_cost)

    added = window.add("assistant", "ok", metadata=metadata)
    snapshot = window.messages()[0]
    for exposed in (metadata, added.metadata, snapshot.metadata):
        details: Any = exposed["reasoning_details"]
        details[0]["data"] = "x" * 1_000

    assert window.wire_transcript() == (expected_wire,)
    assert window.used_tokens == first_cost

    window.add("user", "next")

    [event] = _compaction_events(log)
    assert event.payload["freed_tokens"] == first_cost
    assert window.used_tokens == crude_token_count("next")


def test_zero_cost_history_under_pressure_rejects_cleanly() -> None:
    costs = {"nothing": 0, "heavy": 6}
    window, log, *_ = _make_window(max_tokens=5, costs=costs)

    window.add("user", "nothing")
    window.add("user", "nothing")
    assert window.used_tokens == 0

    with pytest.raises(ValueError, match="window holds"):
        window.add("user", "heavy")

    assert len(log) == 0
    assert len(window.messages()) == 2
    assert window.used_tokens == 0
    assert window.compaction_count == 0


# --- rejected adds are absolutely side-effect free ------------------------------


def test_do_nothing_policy_leaves_zero_events_and_intact_state() -> None:
    def noop(
        messages: Sequence[Message],
        tokens_to_free: int,
        incoming_tokens: int,
        token_counter: Callable[[Message], int],
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


def test_policy_frees_then_stalls_leaves_zero_events_and_intact_state() -> None:
    calls: list[int] = []

    def one_shot_then_noop(
        messages: Sequence[Message],
        tokens_to_free: int,
        incoming_tokens: int,
        token_counter: Callable[[Message], int],
    ) -> list[Message]:
        calls.append(tokens_to_free)
        if len(calls) == 1:
            return list(messages[1:])
        return list(messages)

    costs = {"a": 1, "b": 3, "c": 2}
    window, log, *_ = _make_window(max_tokens=4, costs=costs, policy=one_shot_then_noop)
    window.add("user", "a")
    window.add("user", "b")

    # Needing 2 freed tokens, round 1 drops "a" but frees only 1, forcing
    # round 2, which stalls. The whole add must abort: zero events, zero
    # state change.
    with pytest.raises(ValueError, match="freed no messages"):
        window.add("user", "c")

    assert [m.content for m in window.messages()] == ["a", "b"]
    assert window.used_tokens == 4
    assert window.compaction_count == 0
    assert len(log) == 0


def test_smuggler_policy_rejected_with_zero_events_and_intact_state() -> None:
    def smuggler(
        messages: Sequence[Message],
        tokens_to_free: int,
        incoming_tokens: int,
        token_counter: Callable[[Message], int],
    ) -> list[Message]:
        return [Message(seq=999, role="ghost", content="boo")]

    costs = {"a": 2, "b": 2}
    window, log, *_ = _make_window(max_tokens=3, costs=costs, policy=smuggler)
    window.add("user", "a")

    with pytest.raises(ValueError, match="unknown message seq"):
        window.add("user", "b")

    assert [m.content for m in window.messages()] == ["a"]
    assert window.used_tokens == 2
    assert window.compaction_count == 0
    assert len(log) == 0


def test_padded_superset_policy_rejected_with_zero_events_and_intact_state() -> None:
    def padder(
        messages: Sequence[Message],
        tokens_to_free: int,
        incoming_tokens: int,
        token_counter: Callable[[Message], int],
    ) -> list[Message]:
        return [*messages, messages[0]]

    costs = {"a": 2, "b": 2}
    window, log, *_ = _make_window(max_tokens=3, costs=costs, policy=padder)
    window.add("user", "a")

    with pytest.raises(ValueError, match="duplicated or reordered"):
        window.add("user", "b")

    assert [m.content for m in window.messages()] == ["a"]
    assert window.used_tokens == 2
    assert window.compaction_count == 0
    assert len(log) == 0


def test_stingy_policy_buffers_two_rounds_then_commits_atomically() -> None:
    def stingy(
        messages: Sequence[Message],
        tokens_to_free: int,
        incoming_tokens: int,
        token_counter: Callable[[Message], int],
    ) -> list[Message]:
        return list(messages[1:]) if messages else []

    costs = {"x": 1, "yy": 2}
    window, log, clock, cycles = _make_window(max_tokens=3, costs=costs, policy=stingy)
    for _ in range(3):
        window.add("user", "x")
    clock.advance_us(250)
    cycles.advance()

    # The incoming message needs 2 freed tokens; one stingy drop frees only
    # 1, so the window buffers a second round before committing atomically.
    fourth = window.add("user", "yy")

    assert fourth.seq == 3
    assert [m.content for m in window.messages()] == ["x", "yy"]
    assert [m.seq for m in window.messages()] == [2, 3]
    assert window.used_tokens == 3
    assert window.compaction_count == 2
    events = _compaction_events(log)
    assert len(events) == 2
    assert [event.payload["dropped_seq"] for event in events] == [[0], [1]]
    assert [event.payload["freed_tokens"] for event in events] == [1, 1]
    assert all(event.cycle == 1 and event.t_us == 250 for event in events)


# --- built-in policies as units -------------------------------------------------


def _four_messages() -> list[Message]:
    contents = ("a", "b", "c", "d")
    return [Message(seq=index, role="user", content=c) for index, c in enumerate(contents)]


def test_drop_oldest_pops_front_until_enough_freed() -> None:
    counter = _fixed_counter({"a": 4, "b": 2, "c": 2, "d": 1})

    kept = drop_oldest(_four_messages(), 5, 2, lambda message: counter(message.content))

    assert [m.seq for m in kept] == [2, 3]


def test_drop_oldest_half_cuts_half_then_falls_back_to_front_popping() -> None:
    counter = _fixed_counter({"a": 4, "b": 2, "c": 2, "d": 1})
    messages = _four_messages()

    # Half cut frees a(4)+b(2)=6 >= 5: exactly the oldest half goes.
    kept = drop_oldest_half(messages, 5, 2, lambda message: counter(message.content))
    assert [m.seq for m in kept] == [2, 3]

    # Needing 9 exceeds the half cut (6): c(2) then d(1) are popped too.
    kept = drop_oldest_half(messages, 9, 2, lambda message: counter(message.content))
    assert kept == []


def test_drop_oldest_half_on_odd_count_cuts_floor_half() -> None:
    counter = _fixed_counter({"a": 2, "b": 2, "c": 2})
    messages = [
        Message(seq=index, role="user", content=c) for index, c in enumerate(("a", "b", "c"))
    ]

    kept = drop_oldest_half(messages, 2, 2, lambda message: counter(message.content))

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
    assert window_a.transcript() == window_b.transcript()
    assert window_a.used_tokens == window_b.used_tokens
    assert window_a.compaction_count == window_b.compaction_count == 1
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    log_a.write_jsonl(path_a)
    log_b.write_jsonl(path_b)
    assert path_a.read_bytes() == path_b.read_bytes()
