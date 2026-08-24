"""OpenAI adapter: thin wrapper over the vendor chat completions client."""

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


def _import_openai_sdk() -> Any:
    """Import the vendor SDK lazily; module-level so tests can substitute it."""
    return importlib.import_module("openai")


class OpenAIAdapter:
    """Calls client.chat.completions.create and normalizes the reply.

    Supported model set: chat-completions models that accept
    max_completion_tokens (current reasoning-family models require it; the
    legacy max_tokens parameter is deprecated and never sent). The vendor SDK
    is imported only when no client was injected and the first completion
    runs, so importing this module never pulls the SDK in. Every client
    failure surfaces as AdapterError; retries are out of scope.
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
            sdk = _import_openai_sdk()
            return sdk.OpenAI()
        except AdapterError:
            raise
        except ImportError as exc:
            msg = (
                "OpenAIAdapter requires the 'openai' package; "
                "install it with: pip install awarebench[openai]"
            )
            raise AdapterError(msg) from exc
        except Exception as exc:
            msg = f"failed to construct default OpenAI client: {type(exc).__name__}: {exc}"
            raise AdapterError(msg) from exc

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        """Run one chat completion call and map the reply onto AdapterResponse."""
        client = self._ensure_client()
        request: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            request["max_completion_tokens"] = max_tokens
        try:
            response = client.chat.completions.create(**request)
        except AdapterError:
            raise
        except Exception as exc:
            msg = f"openai chat.completions.create failed: {type(exc).__name__}: {exc}"
            raise AdapterError(msg) from exc
        return self._map_response(response)

    @staticmethod
    def _map_response(response: Any) -> AdapterResponse:
        """Extract text, usage, and stop_reason; malformed shapes raise AdapterError."""
        choices = _require_attr(response, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            msg = "malformed openai response: choices must be a non-empty list"
            raise AdapterError(msg)
        choice = choices[0]
        message = _require_attr(choice, "message")
        text = _require_attr(message, "content")
        if not isinstance(text, str):
            msg = "malformed openai response: message.content must be a str"
            raise AdapterError(msg)
        usage = _require_attr(response, "usage")
        return AdapterResponse(
            text=text,
            reasoning=None,
            prompt_tokens=_require_token(
                _require_attr(usage, "prompt_tokens"), "usage.prompt_tokens"
            ),
            completion_tokens=_require_token(
                _require_attr(usage, "completion_tokens"), "usage.completion_tokens"
            ),
            stop_reason=_normalize_stop_reason(_require_attr(choice, "finish_reason")),
            model=_optional_str(response, "model"),
            request_id=_optional_str(response, "id"),
        )
