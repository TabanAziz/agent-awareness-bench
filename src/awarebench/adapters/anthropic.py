"""Anthropic adapter: thin wrapper over the vendor Messages API client."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from awarebench.adapters.base import (
    AdapterError,
    AdapterResponse,
    _normalize_stop_reason,
    _optional_str,
    _require_attr,
    _require_token,
)


def _import_anthropic_sdk() -> Any:
    """Import the vendor SDK lazily; module-level so tests can substitute it."""
    return importlib.import_module("anthropic")


class AnthropicAdapter:
    """Calls client.messages.create and normalizes the reply.

    The vendor SDK is imported only when no client was injected and the first
    completion runs, so importing this module never pulls the SDK in. Every
    client failure surfaces as AdapterError; retries are out of scope.
    """

    def __init__(self, model: str, client: Any = None) -> None:
        self._model = model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self._default_client()
        return self._client

    def _default_client(self) -> Any:
        try:
            sdk = _import_anthropic_sdk()
            return sdk.Anthropic()
        except AdapterError:
            raise
        except ImportError as exc:
            msg = (
                "AnthropicAdapter requires the 'anthropic' package; "
                "install it with: pip install awarebench[anthropic]"
            )
            raise AdapterError(msg) from exc
        except Exception as exc:
            msg = f"failed to construct default Anthropic client: {type(exc).__name__}: {exc}"
            raise AdapterError(msg) from exc

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        """Run one Messages API call and map the reply onto AdapterResponse.

        Incoming messages are OpenAI-style (see ModelAdapter); role=="system"
        entries are lifted out and joined with blank lines into the top-level
        system parameter the Anthropic API requires.
        """
        client = self._ensure_client()
        system_parts = [message["content"] for message in messages if message["role"] == "system"]
        chat_messages = [message for message in messages if message["role"] != "system"]
        request: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "temperature": temperature,
        }
        if system_parts:
            request["system"] = "\n\n".join(system_parts)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        try:
            response = client.messages.create(**request)
        except AdapterError:
            raise
        except Exception as exc:
            msg = f"anthropic messages.create failed: {type(exc).__name__}: {exc}"
            raise AdapterError(msg) from exc
        return self._map_response(response)

    @staticmethod
    def _map_response(response: Any) -> AdapterResponse:
        """Extract text, usage, and stop_reason; malformed shapes raise AdapterError."""
        content = _require_attr(response, "content")
        if not isinstance(content, (list, tuple)):
            msg = "malformed anthropic response: content must be a list of blocks"
            raise AdapterError(msg)
        parts: list[str] = []
        for block in content:
            candidate = getattr(block, "text", None)
            if isinstance(candidate, str):
                parts.append(candidate)
        text = "\n".join(parts)
        if not text:
            msg = "malformed anthropic response: no content block carries text"
            raise AdapterError(msg)
        usage = _require_attr(response, "usage")
        return AdapterResponse(
            text=text,
            reasoning=None,
            prompt_tokens=_require_token(
                _require_attr(usage, "input_tokens"), "usage.input_tokens"
            ),
            completion_tokens=_require_token(
                _require_attr(usage, "output_tokens"), "usage.output_tokens"
            ),
            stop_reason=_normalize_stop_reason(_require_attr(response, "stop_reason")),
            model=_optional_str(response, "model"),
            request_id=_optional_str(response, "id"),
        )
