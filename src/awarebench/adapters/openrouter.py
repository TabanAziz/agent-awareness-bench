"""OpenRouter adapter over its OpenAI-compatible chat completion endpoint."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from awarebench.adapters.base import (
    AdapterError,
    AdapterResponse,
    _normalize_stop_reason,
    _require_token,
)

OPENROUTER_ENDPOINT: Final[str] = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_S: Final[float] = 120.0

Transport = Callable[[Request, float], bytes]


def _default_transport(request: Request, timeout: float) -> bytes:
    """Send one request and return its raw response body."""
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not isinstance(body, bytes):
        raise AdapterError("openrouter transport returned a non-bytes response")
    return body


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AdapterError(f"malformed openrouter response: {label} must be an object")
    return value


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise AdapterError(f"malformed openrouter response: missing {label}.{key}")
    return mapping[key]


def _optional_str(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


class OpenRouterAdapter:
    """Call OpenRouter and normalize one chat completion into AdapterResponse."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        transport: Transport | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._model = model
        self._api_key = api_key
        self._transport = transport if transport is not None else _default_transport
        self._timeout_s = timeout_s

    def _resolve_api_key(self) -> str:
        key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise AdapterError("OpenRouterAdapter requires OPENROUTER_API_KEY")
        return key

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        """Send one request and map the response without exposing credentials."""
        body: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        request = Request(
            OPENROUTER_ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._resolve_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            raw_response = self._transport(request, self._timeout_s)
        except HTTPError as exc:
            raise AdapterError(f"openrouter request failed with HTTP {exc.code}") from exc
        except Exception as exc:
            raise AdapterError(f"openrouter request failed: {type(exc).__name__}") from exc
        try:
            payload = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError("openrouter response was not valid JSON") from exc
        return self._map_response(payload)

    @staticmethod
    def _map_response(payload: Any) -> AdapterResponse:
        root = _mapping(payload, "root")
        choices = _required(root, "choices", "root")
        if not isinstance(choices, list) or not choices:
            raise AdapterError("malformed openrouter response: choices must be a non-empty list")
        choice = _mapping(choices[0], "choices[0]")
        message = _mapping(_required(choice, "message", "choices[0]"), "message")
        text = _required(message, "content", "message")
        if not isinstance(text, str):
            raise AdapterError("malformed openrouter response: message.content must be a str")
        reasoning = _optional_str(message, "reasoning")
        usage = _mapping(_required(root, "usage", "root"), "usage")
        return AdapterResponse(
            text=text,
            reasoning=reasoning,
            prompt_tokens=_require_token(
                _required(usage, "prompt_tokens", "usage"),
                "usage.prompt_tokens",
            ),
            completion_tokens=_require_token(
                _required(usage, "completion_tokens", "usage"),
                "usage.completion_tokens",
            ),
            stop_reason=_normalize_stop_reason(choice.get("finish_reason")),
            model=_optional_str(root, "model"),
            request_id=_optional_str(root, "id"),
        )
