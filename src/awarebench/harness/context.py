"""Context window simulation with silent compaction.

ContextWindow bounds an agent transcript to max_tokens. When an incoming
message would overflow the window, the configured DropPolicy selects the kept
messages; the dropped ones vanish silently from the agent's view while a
truthful COMPACTION event records their seqs for scoring. Silently losing an
early constraint message under the default policy is a spec requirement: the
probe class C scenarios depend on it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from awarebench.adapters.base import message_token_text
from awarebench.events import EventLog, EventType, JsonValue
from awarebench.harness._validation import (
    require_strict_non_negative_int,
    require_strict_positive_int,
)
from awarebench.harness.clock import CycleCounter, VirtualClock


class Message(BaseModel):
    """One immutable transcript entry.

    seq is scoring-side identity assigned by ContextWindow; it is never
    surfaced agent-visible. Read transcripts through
    ContextWindow.transcript(), which exposes only (role, content) pairs, or
    wire_transcript(), which also includes replay metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: StrictInt = Field(ge=0)
    role: str
    content: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("role")
    @classmethod
    def _role_must_be_non_empty(cls, value: str) -> str:
        """Reject empty roles; they carry no protocol meaning."""
        if not value:
            raise ValueError("role must be non-empty")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_must_not_replace_core_fields(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        overlap = {"role", "content"}.intersection(value)
        if overlap:
            raise ValueError(f"metadata cannot replace core fields: {sorted(overlap)}")
        return value


def crude_token_count(content: str) -> int:
    """Crude token approximation: one token per four characters.

    Deterministic and dependency-free placeholder until real tokenizer counts
    arrive with the adapter layer.
    """
    return len(content) // 4


type MessageTokenCounter = Callable[[Message], int]
type DropPolicy = Callable[[Sequence[Message], int, int, MessageTokenCounter], list[Message]]
"""Selects survivors under pressure.

Called as policy(messages, tokens_to_free, incoming_tokens, message_counter)
with the window's current messages in order. Must return an ordered subset of
those messages, the kept ones, without mutating the input and without
inventing entries; the window validates that shape before anything is logged
or mutated, then derives the dropped set by seq difference.

The counter argument prices a complete Message using the window's effective
per-string tokenizer over content plus canonical replay metadata. Vendor
usage aggregates never flow through it; they route solely into
BudgetAccountant. Policies must size messages under this one ledger so their
decisions match the window's accounting.
"""


def drop_oldest(
    messages: Sequence[Message],
    tokens_to_free: int,
    incoming_tokens: int,
    message_counter: MessageTokenCounter,
) -> list[Message]:
    """Pop messages from the front until enough tokens are freed or empty."""
    kept = list(messages)
    freed = 0
    while kept and freed < tokens_to_free:
        freed += message_counter(kept.pop(0))
    return kept


def drop_oldest_half(
    messages: Sequence[Message],
    tokens_to_free: int,
    incoming_tokens: int,
    message_counter: MessageTokenCounter,
) -> list[Message]:
    """Drop the oldest half in one cut; front-pop more if that falls short."""
    kept = list(messages)
    cut = len(kept) // 2
    freed = 0
    for message in kept[:cut]:
        freed += message_counter(message)
    kept = kept[cut:]
    while kept and freed < tokens_to_free:
        freed += message_counter(kept.pop(0))
    return kept


class ContextWindow:
    """Token-bounded transcript that silently compacts under pressure.

    Every add assigns a fresh monotonic seq. When the incoming message would
    push usage past max_tokens, the DropPolicy picks the survivors; each
    compaction round logs one truthful COMPACTION event carrying the dropped
    seqs and freed token count, stamped with the current virtual time and
    cycle. dropped_seq names transcript-scoped Message.seq values, never the
    event log's own seq namespace.

    Rejected adds are absolutely side-effect free: domain rejections
    (oversized message, bad input, a policy that frees nothing or returns a
    malformed kept list) raise ValueError with zero events logged and zero
    state change. Compaction rounds are buffered and committed atomically
    only once the incoming message is known to fit.

    Token accounting contract: token_counter is a per-string tokenizer
    function (tiktoken-style). Each message is charged for its content plus a
    canonical JSON serialization of replay metadata. Vendor-reported usage
    aggregates never flow through it; they route solely into
    BudgetAccountant. The window measures perceived pressure under this one
    consistent ledger.
    """

    def __init__(
        self,
        event_log: EventLog,
        clock: VirtualClock,
        cycles: CycleCounter,
        max_tokens: int,
        token_counter: Callable[[str], int] | None = None,
        policy: DropPolicy | None = None,
    ) -> None:
        require_strict_positive_int("max_tokens", max_tokens)
        self._log = event_log
        self._clock = clock
        self._cycles = cycles
        self._max_tokens = max_tokens
        self._token_counter: Callable[[str], int] = (
            token_counter if token_counter is not None else crude_token_count
        )
        self._policy: DropPolicy = policy if policy is not None else drop_oldest
        self._messages: list[Message] = []
        self._used_tokens = 0
        self._compaction_count = 0
        self._next_seq = 0

    def add(
        self,
        role: str,
        content: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> Message:
        """Add a message, compacting silently when the window overflows."""
        message = self._candidate(role, content, metadata)
        incoming, rounds, working, used = self._plan_add(message)
        for kept, dropped, freed in rounds:
            self._log.append(
                EventType.COMPACTION,
                self._cycles.current,
                self._clock.now_us,
                {
                    "dropped_seq": [message.seq for message in dropped],
                    "freed_tokens": freed,
                },
            )
            self._compaction_count += 1
        self._messages = working
        self._used_tokens = used
        self._next_seq += 1
        self._messages.append(message)
        self._used_tokens += incoming
        return message.model_copy(deep=True)

    def validate_add(
        self,
        role: str,
        content: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        """Validate and plan an add without mutating state or logging events."""
        self._plan_add(self._candidate(role, content, metadata))

    def _candidate(
        self,
        role: str,
        content: str,
        metadata: dict[str, JsonValue] | None,
    ) -> Message:
        if not isinstance(role, str) or not role:
            raise ValueError(f"role must be a non-empty str, got {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"content must be a str, got {type(content).__name__}")
        return Message(
            seq=self._next_seq,
            role=role,
            content=content,
            metadata={} if metadata is None else copy.deepcopy(metadata),
        )

    def _message_tokens(self, message: Message) -> int:
        tokens = self._token_counter(message_token_text(message.content, message.metadata))
        require_strict_non_negative_int("token count", tokens)
        return tokens

    def _plan_add(
        self, message: Message
    ) -> tuple[int, list[tuple[list[Message], list[Message], int]], list[Message], int]:
        incoming = self._message_tokens(message)
        require_strict_non_negative_int("token count", incoming)
        if incoming > self._max_tokens:
            raise ValueError(
                f"incoming message needs {incoming} tokens, window holds {self._max_tokens}"
            )
        rounds: list[tuple[list[Message], list[Message], int]] = []
        working = list(self._messages)
        used = self._used_tokens
        while used + incoming > self._max_tokens:
            policy_messages = tuple(message.model_copy(deep=True) for message in working)
            selected = list(
                self._policy(
                    policy_messages,
                    used + incoming - self._max_tokens,
                    incoming,
                    self._message_tokens,
                )
            )
            self._validate_kept(working, selected)
            kept_seqs = {message.seq for message in selected}
            kept = [message for message in working if message.seq in kept_seqs]
            dropped = [message for message in working if message.seq not in kept_seqs]
            if not dropped:
                raise ValueError("drop policy freed no messages; cannot make room")
            freed = sum(self._message_tokens(message) for message in dropped)
            rounds.append((kept, dropped, freed))
            working = kept
            used -= freed
        return incoming, rounds, working, used

    def messages(self) -> tuple[Message, ...]:
        """Return the surviving transcript in order, as a defensive copy."""
        return tuple(message.model_copy(deep=True) for message in self._messages)

    def transcript(self) -> tuple[tuple[str, str], ...]:
        """Agent-safe view: (role, content) pairs without scoring-side seqs."""
        return tuple((message.role, message.content) for message in self._messages)

    def wire_transcript(self) -> tuple[dict[str, JsonValue], ...]:
        """Agent-safe wire messages, including replay metadata but no scoring seqs."""
        result: list[dict[str, JsonValue]] = []
        for message in self._messages:
            wire_message: dict[str, JsonValue] = {
                "role": message.role,
                "content": message.content,
            }
            wire_message.update(copy.deepcopy(message.metadata))
            result.append(wire_message)
        return tuple(result)

    @property
    def used_tokens(self) -> int:
        """Token cost of the surviving messages under the injected counter."""
        return self._used_tokens

    @property
    def compaction_count(self) -> int:
        """Number of committed compaction rounds so far."""
        return self._compaction_count

    @staticmethod
    def _validate_kept(input_messages: Sequence[Message], kept: list[Message]) -> None:
        """Reject kept lists that are not ordered subsequences of the input."""
        index_by_seq = {message.seq: index for index, message in enumerate(input_messages)}
        last_index = -1
        for message in kept:
            index = index_by_seq.get(message.seq)
            if index is None:
                raise ValueError(f"drop policy returned unknown message seq {message.seq}")
            if index <= last_index:
                raise ValueError("drop policy returned duplicated or reordered messages")
            last_index = index
