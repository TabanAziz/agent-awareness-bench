"""OpenRouter adapter over its OpenAI-compatible chat completion endpoint."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from awarebench.adapters.base import (
    AdapterError,
    AdapterResponse,
    _normalize_stop_reason,
    _require_token,
)

OPENROUTER_ENDPOINT: Final[str] = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_S: Final[float] = 120.0
MAX_RESPONSE_BYTES: Final[int] = 8 * 1024 * 1024

Transport = Callable[[Request, float], bytes]


class _RejectRedirects(HTTPRedirectHandler):
    """Keep bearer credentials on the one configured OpenRouter origin."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Any:
        del req, fp, code, msg, headers, newurl


def _default_transport(request: Request, timeout: float) -> bytes:
    """Send one request and return its raw response body."""
    opener = build_opener(_RejectRedirects())
    with opener.open(request, timeout=timeout) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
    if not isinstance(body, bytes):
        raise AdapterError("openrouter transport returned a non-bytes response")
    if len(body) > MAX_RESPONSE_BYTES:
        raise AdapterError("openrouter response exceeded size limit")
    return body


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AdapterError(f"malformed openrouter response: {label} must be an object")
    return value


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise AdapterError(f"malformed openrouter response: missing {label}.{key}")
    return mapping[key]


def _required_str(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _required(mapping, key, label)
    if not isinstance(value, str) or not value:
        raise AdapterError(f"malformed openrouter response: {label}.{key} must be a string")
    return value


def _reasoning_text(message: Mapping[str, Any]) -> str | None:
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise AdapterError("malformed openrouter response: message.reasoning must be a string")
        return reasoning
    details = message.get("reasoning_details")
    if details is None:
        return None
    if not isinstance(details, list):
        raise AdapterError(
            "malformed openrouter response: message.reasoning_details must be a list"
        )
    texts: list[str] = []
    for index, raw_detail in enumerate(details):
        detail = _mapping(raw_detail, f"message.reasoning_details[{index}]")
        detail_type = detail.get("type")
        if detail_type == "reasoning.text":
            value = detail.get("text")
        elif detail_type == "reasoning.summary":
            value = detail.get("summary")
        elif detail_type == "reasoning.encrypted":
            continue
        else:
            raise AdapterError("malformed openrouter response: unknown reasoning detail type")
        if not isinstance(value, str):
            raise AdapterError(
                "malformed openrouter response: reasoning detail content must be a string"
            )
        if value:
            texts.append(value)
    return "\n".join(texts) or None


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
        if _required_str(root, "object", "root") != "chat.completion":
            raise AdapterError("malformed openrouter response: root.object must be chat.completion")
        response_model = _required_str(root, "model", "root")
        request_id = _required_str(root, "id", "root")
        choices = _required(root, "choices", "root")
        if not isinstance(choices, list) or not choices:
            raise AdapterError("malformed openrouter response: choices must be a non-empty list")
        choice = _mapping(choices[0], "choices[0]")
        choice_index = _required(choice, "index", "choices[0]")
        if isinstance(choice_index, bool) or not isinstance(choice_index, int) or choice_index != 0:
            raise AdapterError("malformed openrouter response: choices[0].index must be 0")
        finish_reason = _required_str(choice, "finish_reason", "choices[0]")
        message = _mapping(_required(choice, "message", "choices[0]"), "message")
        if _required_str(message, "role", "message") != "assistant":
            raise AdapterError("malformed openrouter response: message.role must be assistant")
        text = _required(message, "content", "message")
        if not isinstance(text, str):
            raise AdapterError("malformed openrouter response: message.content must be a str")
        reasoning = _reasoning_text(message)
        usage = _mapping(_required(root, "usage", "root"), "usage")
        prompt_tokens = _require_token(
            _required(usage, "prompt_tokens", "usage"),
            "usage.prompt_tokens",
        )
        completion_tokens = _require_token(
            _required(usage, "completion_tokens", "usage"),
            "usage.completion_tokens",
        )
        total_tokens = _require_token(
            _required(usage, "total_tokens", "usage"),
            "usage.total_tokens",
        )
        if total_tokens != prompt_tokens + completion_tokens:
            raise AdapterError(
                "malformed openrouter response: usage.total_tokens does not match components"
            )
        return AdapterResponse(
            text=text,
            reasoning=reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            stop_reason=_normalize_stop_reason(finish_reason),
            model=response_model,
            request_id=request_id,
        )
