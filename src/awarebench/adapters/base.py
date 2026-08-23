"""Adapter contracts: response model, error type, and the ModelAdapter protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class AdapterResponse(BaseModel):
    """Single normalized model reply produced by every adapter.

    Token counts are strict non-negative ints (bools and floats rejected);
    stop_reason is always a non-empty string, with adapters substituting
    "unknown" when the vendor omits it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    reasoning: str | None = None
    prompt_tokens: StrictInt = Field(ge=0)
    completion_tokens: StrictInt = Field(ge=0)
    stop_reason: str = Field(min_length=1)


class AdapterError(Exception):
    """Transport, auth, or API-shape failure raised inside an adapter.

    Converting failures into logged events is the agent loop's responsibility
    (it lands later), never the adapter's: adapters raise, the loop records.
    """


@runtime_checkable
class ModelAdapter(Protocol):
    """Anything that can run one chat completion over plain message dicts."""

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        """Return one AdapterResponse for the given {"role"/"content"} messages."""
        ...


_MISSING: Final = object()


def _require_attr(container: Any, name: str) -> Any:
    """Return container.name, raising AdapterError when the attribute is absent."""
    value = getattr(container, name, _MISSING)
    if value is _MISSING:
        msg = f"malformed SDK response: missing attribute {name!r}"
        raise AdapterError(msg)
    return value


def _require_token(value: Any, label: str) -> int:
    """Return value as a token count, rejecting bools, floats, and negatives."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"malformed SDK response: {label} must be an int"
        raise AdapterError(msg)
    if value < 0:
        msg = f"malformed SDK response: {label} must be >= 0"
        raise AdapterError(msg)
    return value


def _normalize_stop_reason(value: Any) -> str:
    """Map a vendor stop marker onto a non-empty string, defaulting to "unknown"."""
    return value if isinstance(value, str) and value else "unknown"
