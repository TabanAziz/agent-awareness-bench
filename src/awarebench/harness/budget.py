"""Budget accounting for tokens, tool calls, and wall time."""

from __future__ import annotations


class BudgetAccountant:
    """Accumulates token, tool-call, and wall-time usage for one run."""

    def __init__(self) -> None:
        self._tokens_used: int = 0
        self._tool_calls: int = 0
        self._wall_us_used: int = 0

    def add_tokens(self, n: int) -> None:
        """Add n consumed tokens; n must be >= 0."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self._tokens_used += n

    def add_tool_call(self) -> None:
        """Count one tool call."""
        self._tool_calls += 1

    def add_wall_us(self, us: int) -> None:
        """Add us measured wall-clock microseconds; us must be >= 0."""
        if us < 0:
            raise ValueError(f"us must be >= 0, got {us}")
        self._wall_us_used += us

    @property
    def tokens_used(self) -> int:
        """Total tokens consumed so far."""
        return self._tokens_used

    @property
    def tool_calls(self) -> int:
        """Total number of tool calls so far."""
        return self._tool_calls

    @property
    def wall_us_used(self) -> int:
        """Total wall-clock microseconds accumulated so far."""
        return self._wall_us_used

    def snapshot(self) -> dict[str, int]:
        """Return the three totals as a plain dict."""
        return {
            "tokens_used": self._tokens_used,
            "tool_calls": self._tool_calls,
            "wall_us_used": self._wall_us_used,
        }
