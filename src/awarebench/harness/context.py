"""Context window simulation with silent compaction.

ContextWindow bounds an agent transcript to max_tokens. When an incoming
message would overflow the window, the configured DropPolicy selects the kept
messages; the dropped ones vanish silently from the agent's view while a
truthful COMPACTION event records their seqs for scoring. Silently losing an
early constraint message under the default policy is a spec requirement: the
probe class C scenarios depend on it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from awarebench.events import EventLog, EventType
from awarebench.harness.clock import CycleCounter, VirtualClock


class Message(BaseModel):
    """One immutable transcript entry; seq is assigned by ContextWindow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def _role_must_be_non_empty(cls, value: str) -> str:
        """Reject empty roles; they carry no protocol meaning."""
        if not value:
            raise ValueError("role must be non-empty")
        return value


def crude_token_count(content: str) -> int:
    """Crude token approximation: one token per four characters.

    Deterministic and dependency-free placeholder until real tokenizer counts
    arrive with the adapter layer.
    """
    return len(content) // 4


type DropPolicy = Callable[[Sequence[Message], int, int], list[Message]]
"""Selects survivors under pressure.

Contract: called as policy(messages, tokens_to_free, incoming_tokens) with the
window's current messages in order; must return an ordered subset of those
messages (the kept ones) without mutating the input. The window derives the
dropped set by seq difference and logs it.
"""


def drop_oldest(token_counter: Callable[[str], int] = crude_token_count) -> DropPolicy:
    """Pop messages from the front until enough tokens are freed or empty."""

    def policy(
        messages: Sequence[Message], tokens_to_free: int, incoming_tokens: int
    ) -> list[Message]:
        kept = list(messages)
        freed = 0
        while kept and freed < tokens_to_free:
            freed += token_counter(kept.pop(0).content)
        return kept

    return policy


def drop_oldest_half(
    token_counter: Callable[[str], int] = crude_token_count,
) -> DropPolicy:
    """Drop the oldest half in one cut; front-pop more if that falls short."""

    def policy(
        messages: Sequence[Message], tokens_to_free: int, incoming_tokens: int
    ) -> list[Message]:
        kept = list(messages)
        cut = len(kept) // 2
        freed = 0
        for message in kept[:cut]:
            freed += token_counter(message.content)
        kept = kept[cut:]
        while kept and freed < tokens_to_free:
            freed += token_counter(kept.pop(0).content)
        return kept

    return policy


class ContextWindow:
    """Token-bounded transcript that silently compacts under pressure.

    Every add assigns a fresh monotonic seq. When the incoming message would
    push usage past max_tokens, the DropPolicy picks the survivors; each
    compaction round logs one truthful COMPACTION event carrying the dropped
    seqs and freed token count, stamped with the current virtual time and
    cycle. Domain rejections (oversized message, bad input, do-nothing
    policy) raise ValueError and never touch the log or the message state.
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
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError(f"max_tokens must be an int, got {type(max_tokens).__name__}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        self._log = event_log
        self._clock = clock
        self._cycles = cycles
        self._max_tokens = max_tokens
        self._token_counter: Callable[[str], int] = (
            token_counter if token_counter is not None else crude_token_count
        )
        self._policy: DropPolicy = (
            policy if policy is not None else drop_oldest(self._token_counter)
        )
        self._messages: list[Message] = []
        self._used_tokens = 0
        self._compaction_count = 0
        self._next_seq = 0

    def add(self, role: str, content: str) -> Message:
        """Add a message, compacting silently when the window overflows."""
        if not isinstance(role, str) or not role:
            raise ValueError(f"role must be a non-empty str, got {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"content must be a str, got {type(content).__name__}")
        incoming = self._token_counter(content)
        if incoming < 0:
            raise ValueError(f"token_counter returned negative count: {incoming}")
        if incoming > self._max_tokens:
            raise ValueError(
                f"incoming message needs {incoming} tokens, window holds {self._max_tokens}"
            )
        while self._used_tokens + incoming > self._max_tokens:
            self._compact(
                tokens_to_free=self._used_tokens + incoming - self._max_tokens,
                incoming_tokens=incoming,
            )
        message = Message(seq=self._next_seq, role=role, content=content)
        self._next_seq += 1
        self._messages.append(message)
        self._used_tokens += incoming
        return message

    def messages(self) -> tuple[Message, ...]:
        """Return the surviving transcript in order, as a defensive copy."""
        return tuple(self._messages)

    @property
    def used_tokens(self) -> int:
        """Token cost of the surviving messages under the injected counter."""
        return self._used_tokens

    @property
    def compaction_count(self) -> int:
        """Number of compaction rounds performed so far."""
        return self._compaction_count

    def _compact(self, tokens_to_free: int, incoming_tokens: int) -> None:
        """Run one compaction round: policy, truthful event, state update."""
        kept = self._policy(self._messages, tokens_to_free, incoming_tokens)
        kept_seqs = {message.seq for message in kept}
        dropped = [message for message in self._messages if message.seq not in kept_seqs]
        if not dropped:
            raise ValueError("drop policy freed no messages; cannot make room")
        freed_tokens = sum(self._token_counter(message.content) for message in dropped)
        self._log.append(
            EventType.COMPACTION,
            self._cycles.current,
            self._clock.now_us,
            {
                "dropped_seq": [message.seq for message in dropped],
                "freed_tokens": freed_tokens,
            },
        )
        self._messages = list(kept)
        self._used_tokens -= freed_tokens
        self._compaction_count += 1
