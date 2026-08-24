"""Deterministic scripted adapter for tests and offline runs."""

from __future__ import annotations

from collections.abc import Sequence

from awarebench.adapters.base import AdapterResponse


class StubAdapter:
    """Scripted adapter that pops responses FIFO and repeats the last one forever.

    Zero network, zero vendor imports. Usage uses the documented crude
    len(text) // 4 character approximation in both directions (the prompt side
    sums message contents). temperature is ignored on purpose: determinism
    first.
    """

    def __init__(self, responses: Sequence[str]) -> None:
        if not responses:
            raise ValueError("responses must contain at least one entry")
        self._responses = list(responses)
        self._cursor = 0
        self._calls = 0

    @property
    def call_count(self) -> int:
        """Number of complete() invocations so far."""
        return self._calls

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        """Return the next scripted response; exhausted scripts repeat the last one."""
        del temperature, max_tokens  # ignored on purpose: determinism first
        self._calls += 1
        if self._cursor < len(self._responses):
            text = self._responses[self._cursor]
            self._cursor += 1
        else:
            text = self._responses[-1]
        prompt_chars = sum(len(message["content"]) for message in messages)
        return AdapterResponse(
            text=text,
            reasoning=None,
            prompt_tokens=prompt_chars // 4,
            completion_tokens=len(text) // 4,
            stop_reason="end_turn",
        )
