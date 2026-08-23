"""Budget accounting for tokens, tool calls, and wall time."""

from __future__ import annotations

from awarebench.harness._validation import require_strict_non_negative_int


class BudgetAccountant:
    """Accumulates token, tool-call, and wall-time usage for one run."""

    def __init__(self) -> None:
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._tool_calls: int = 0
        self._wall_us_used: int = 0

    def add_tokens(self, prompt: int, completion: int) -> None:
        """Add consumed tokens split into prompt and completion counts."""
        require_strict_non_negative_int("prompt", prompt)
        require_strict_non_negative_int("completion", completion)
        self._prompt_tokens += prompt
        self._completion_tokens += completion

    def add_tool_call(self) -> None:
        """Count one tool call."""
        self._tool_calls += 1

    def add_wall_us(self, us: int) -> None:
        """Add us measured wall-clock microseconds; us must be a non-negative int."""
        require_strict_non_negative_int("us", us)
        self._wall_us_used += us

    @property
    def prompt_tokens(self) -> int:
        """Total prompt tokens consumed so far."""
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """Total completion tokens consumed so far."""
        return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed so far (prompt + completion)."""
        return self._prompt_tokens + self._completion_tokens

    @property
    def tool_calls(self) -> int:
        """Total number of tool calls so far."""
        return self._tool_calls

    @property
    def wall_us_used(self) -> int:
        """Total wall-clock microseconds accumulated so far."""
        return self._wall_us_used

    def snapshot(self) -> dict[str, int]:
        """Return all totals as a plain dict."""
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self._tool_calls,
            "wall_us_used": self._wall_us_used,
        }
