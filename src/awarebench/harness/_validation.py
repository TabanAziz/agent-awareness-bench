"""Shared strict runtime validation helpers.

All harness modules reject malformed input with ValueError (mirroring pydantic
ValidationError, a ValueError subclass) instead of fragmenting the contract
across TypeError; these helpers keep the strict-int semantics identical.
"""

from __future__ import annotations


def require_strict_non_negative_int(field: str, value: int) -> None:
    """Reject bools and floats like pydantic StrictInt, then negatives."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field} must be >= 0, got {value}")


def require_strict_positive_int(field: str, value: int) -> None:
    """Reject bools and floats like pydantic StrictInt, then values below 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{field} must be >= 1, got {value}")
