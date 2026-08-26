"""Adapter contracts: response model, error type, and the ModelAdapter protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from awarebench.events import JsonValue

type AdapterMessage = dict[str, JsonValue]


class AdapterResponse(BaseModel):
    """Single normalized model reply produced by every adapter.

    Token counts are strict non-negative ints (bools and floats rejected);
    stop_reason is always a non-empty string, with adapters substituting
    "unknown" when the vendor omits it. model and request_id are best-effort
    vendor metadata and stay None when the SDK response omits them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    reasoning: str | None = None
    assistant_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    prompt_tokens: StrictInt = Field(ge=0)
    completion_tokens: StrictInt = Field(ge=0)
    stop_reason: str = Field(min_length=1)
    model: str | None = None
    request_id: str | None = None


class AdapterError(Exception):
    """Transport, auth, or API-shape failure raised inside an adapter.

    Converting failures into logged events is the agent loop's responsibility
    (it lands later), never the adapter's: adapters raise, the loop records.
    """


@runtime_checkable
class ModelAdapter(Protocol):
    """Anything that can run one chat completion over plain message dicts.

    Message lists follow the OpenAI style: {"role": ..., "content": ...} dicts
    where role may be "system". Each adapter translates this common shape into
    its vendor wire format (for example, the Anthropic adapter lifts system
    messages to the top-level system parameter).
    """

    def complete(
        self,
        messages: Sequence[AdapterMessage],
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


def _optional_str(container: Any, name: str) -> str | None:
    """Return container.name when it is a str, else None (tolerant metadata read)."""
    value = getattr(container, name, None)
    return value if isinstance(value, str) else None
