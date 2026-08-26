"""Tests for the adapter layer: stub, vendor wrappers, and the protocol."""

from __future__ import annotations

import json
from email.message import Message
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Self
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from pydantic import ValidationError

import awarebench.adapters.anthropic as anthropic_module
import awarebench.adapters.openai as openai_module
import awarebench.adapters.openrouter as openrouter_module
from awarebench.adapters import (
    AdapterError,
    AdapterResponse,
    AnthropicAdapter,
    ModelAdapter,
    OpenAIAdapter,
    OpenRouterAdapter,
    StubAdapter,
)


class _RecordingEndpoint:
    """Stands in for a vendor SDK endpoint, recording create() kwargs."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _ExplodingEndpoint:
    """Stands in for a vendor SDK endpoint whose transport always fails."""

    def create(self, **kwargs: Any) -> Any:
        raise RuntimeError("transport exploded")


class _RecordingTransport:
    """Records an OpenRouter request and returns one raw response body."""

    def __init__(self, response: dict[str, Any] | bytes) -> None:
        self._response = (
            json.dumps(response).encode("utf-8") if isinstance(response, dict) else response
        )
        self.calls: list[tuple[Request, float]] = []

    def __call__(self, request: Request, timeout: float) -> bytes:
        self.calls.append((request, timeout))
        return self._response


class _ExplodingTransport:
    """OpenRouter transport whose request always fails."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __call__(self, request: Request, timeout: float) -> bytes:
        del request, timeout
        raise self._error


def _anthropic_response(
    text: str = "hello",
    stop_reason: str | None = "end_turn",
    input_tokens: int = 11,
    output_tokens: int = 7,
    model: str | None = "claude-x",
    response_id: str | None = "msg_1",
) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
        model=model,
        id=response_id,
    )


def _openai_response(
    text: str = "hello",
    finish_reason: str | None = "stop",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    model: str | None = "gpt-x",
    response_id: str | None = "cmpl_1",
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        model=model,
        id=response_id,
    )


def _openrouter_response(
    text: str = "hello",
    reasoning: str | None = "checked the trace",
    finish_reason: str | None = "stop",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    model: str | None = "vendor/model",
    response_id: str | None = "gen_1",
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "reasoning": reasoning},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "object": "chat.completion",
        "model": model,
        "id": response_id,
    }


# --- stub adapter --------------------------------------------------------


def test_stub_pops_script_fifo_then_repeats_last_forever() -> None:
    adapter = StubAdapter(["first", "second"])

    texts = [adapter.complete([{"role": "user", "content": "m"}]).text for _ in range(4)]

    assert texts == ["first", "second", "second", "second"]
    assert adapter.call_count == 4


def test_stub_call_count_starts_at_zero() -> None:
    assert StubAdapter(["only"]).call_count == 0


def test_stub_usage_uses_documented_len_over_four_approximation() -> None:
    adapter = StubAdapter(["abcdefghij"])  # 10 chars -> 2 completion tokens

    response = adapter.complete(
        [
            {"role": "system", "content": "12345678"},  # 8 chars
            {"role": "user", "content": "12"},  # 2 chars
        ]
    )

    assert response.prompt_tokens == 2  # (8 + 2) // 4
    assert response.completion_tokens == 2


def test_stub_ignores_temperature_for_determinism() -> None:
    adapter = StubAdapter(["same"])

    cold = adapter.complete([{"role": "user", "content": "x"}], temperature=0.0)
    hot = adapter.complete([{"role": "user", "content": "x"}], temperature=1.7)

    assert cold == hot


def test_stub_rejects_empty_script() -> None:
    with pytest.raises(ValueError, match="at least one"):
        StubAdapter([])


def test_stub_leaves_model_and_request_id_unset() -> None:
    adapter = StubAdapter(["x"])

    response = adapter.complete([{"role": "user", "content": "y"}])

    assert response.model is None
    assert response.request_id is None


# --- AdapterResponse validation -----------------------------------------


def test_adapter_response_accepts_zero_tokens_and_reasoning_default() -> None:
    response = AdapterResponse(text="ok", prompt_tokens=0, completion_tokens=0, stop_reason="stop")

    assert response.reasoning is None
    assert response.model is None
    assert response.request_id is None


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("prompt_tokens", -1),
        ("completion_tokens", -1),
        ("prompt_tokens", True),
        ("completion_tokens", True),
        ("prompt_tokens", 2.5),
        ("completion_tokens", 2.5),
    ],
)
def test_adapter_response_rejects_negative_bool_and_float_tokens(
    field: str, bad_value: Any
) -> None:
    values: dict[str, Any] = {
        "text": "ok",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "stop_reason": "stop",
    }
    values[field] = bad_value

    with pytest.raises(ValidationError):
        AdapterResponse(**values)


def test_adapter_response_rejects_empty_stop_reason() -> None:
    with pytest.raises(ValidationError):
        AdapterResponse(text="ok", prompt_tokens=1, completion_tokens=1, stop_reason="")


def test_adapter_response_is_frozen() -> None:
    response = AdapterResponse(text="ok", prompt_tokens=1, completion_tokens=1, stop_reason="stop")

    with pytest.raises(ValidationError):
        response.text = "mutated"


# --- protocol conformance ------------------------------------------------


def test_all_adapters_satisfy_model_adapter_protocol() -> None:
    stub: ModelAdapter = StubAdapter(["x"])
    anthropic_adapter: ModelAdapter = AnthropicAdapter(model="claude-x")
    openai_adapter: ModelAdapter = OpenAIAdapter(model="gpt-x")
    openrouter_adapter: ModelAdapter = OpenRouterAdapter(
        model="vendor/model",
        api_key="test-key",
        transport=_RecordingTransport(_openrouter_response()),
    )

    assert isinstance(stub, ModelAdapter)
    assert isinstance(anthropic_adapter, ModelAdapter)
    assert isinstance(openai_adapter, ModelAdapter)
    assert isinstance(openrouter_adapter, ModelAdapter)


# --- anthropic adapter ---------------------------------------------------


def test_anthropic_maps_fields_exactly() -> None:
    endpoint = _RecordingEndpoint(_anthropic_response())
    adapter = AnthropicAdapter(model="claude-x", client=SimpleNamespace(messages=endpoint))

    response = adapter.complete([{"role": "user", "content": "hi"}], temperature=0.4, max_tokens=64)

    assert response == AdapterResponse(
        text="hello",
        reasoning=None,
        prompt_tokens=11,
        completion_tokens=7,
        stop_reason="end_turn",
        model="claude-x",
        request_id="msg_1",
    )
    assert endpoint.calls == [
        {
            "model": "claude-x",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.4,
            "max_tokens": 64,
        }
    ]
    assert "system" not in endpoint.calls[0]


def test_anthropic_concatenates_all_text_blocks() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(text="first"),
            SimpleNamespace(flags=1),  # non-text block interleaved
            SimpleNamespace(text="second"),
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        stop_reason="end_turn",
        model="claude-x",
        id="msg_1",
    )
    endpoint = _RecordingEndpoint(response)
    adapter = AnthropicAdapter(model="claude-x", client=SimpleNamespace(messages=endpoint))

    result = adapter.complete([{"role": "user", "content": "hi"}])

    assert result.text == "first\nsecond"


def test_anthropic_lifts_system_messages_to_top_level() -> None:
    endpoint = _RecordingEndpoint(_anthropic_response())
    adapter = AnthropicAdapter(model="claude-x", client=SimpleNamespace(messages=endpoint))

    adapter.complete(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "never lie"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "go on"},
        ]
    )

    call = endpoint.calls[0]
    assert call["system"] == "be terse\n\nnever lie"
    assert call["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "go on"},
    ]


def test_anthropic_omits_max_tokens_when_not_requested() -> None:
    endpoint = _RecordingEndpoint(_anthropic_response())
    adapter = AnthropicAdapter(model="claude-x", client=SimpleNamespace(messages=endpoint))

    adapter.complete([{"role": "user", "content": "hi"}])

    assert "max_tokens" not in endpoint.calls[0]


@pytest.mark.parametrize("stop_reason", [None, ""])
def test_anthropic_falls_back_to_unknown_stop_reason(stop_reason: str | None) -> None:
    endpoint = _RecordingEndpoint(_anthropic_response(stop_reason=stop_reason))
    adapter = AnthropicAdapter(model="claude-x", client=SimpleNamespace(messages=endpoint))

    assert adapter.complete([]).stop_reason == "unknown"


@pytest.mark.parametrize(
    "broken_response",
    [
        SimpleNamespace(),  # content missing
        SimpleNamespace(content="hello", usage=None, stop_reason="end_turn"),  # not block list
        SimpleNamespace(  # no block carries text
            content=[SimpleNamespace(flags=1)], usage=None, stop_reason="end_turn"
        ),
        SimpleNamespace(content=[SimpleNamespace(text="hello")], stop_reason="end_turn"),
        SimpleNamespace(  # input_tokens missing
            content=[SimpleNamespace(text="hello")],
            usage=SimpleNamespace(output_tokens=7),
            stop_reason="end_turn",
        ),
        SimpleNamespace(  # negative token count
            content=[SimpleNamespace(text="hello")],
            usage=SimpleNamespace(input_tokens=-1, output_tokens=7),
            stop_reason="end_turn",
        ),
        SimpleNamespace(  # bool token count
            content=[SimpleNamespace(text="hello")],
            usage=SimpleNamespace(input_tokens=True, output_tokens=7),
            stop_reason="end_turn",
        ),
    ],
)
def test_anthropic_malformed_shapes_raise_adapter_error(broken_response: Any) -> None:
    endpoint = _RecordingEndpoint(broken_response)
    adapter = AnthropicAdapter(model="claude-x", client=SimpleNamespace(messages=endpoint))

    with pytest.raises(AdapterError, match="malformed"):
        adapter.complete([{"role": "user", "content": "hi"}])


def test_anthropic_wraps_client_failures_preserving_cause() -> None:
    adapter = AnthropicAdapter(
        model="claude-x", client=SimpleNamespace(messages=_ExplodingEndpoint())
    )

    with pytest.raises(AdapterError, match="anthropic") as excinfo:
        adapter.complete([{"role": "user", "content": "hi"}])

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "transport exploded"


def test_anthropic_without_client_names_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error() -> Any:
        raise ImportError("simulated missing anthropic sdk")

    monkeypatch.setattr(anthropic_module, "_import_anthropic_sdk", _raise_import_error)
    adapter = AnthropicAdapter(model="claude-x")

    with pytest.raises(AdapterError, match=r"awarebench\[anthropic\]") as excinfo:
        adapter.complete([{"role": "user", "content": "hi"}])

    assert isinstance(excinfo.value.__cause__, ImportError)


def test_anthropic_wraps_non_import_error_from_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_runtime_error() -> Any:
        raise RuntimeError("sdk exploded on import")

    monkeypatch.setattr(anthropic_module, "_import_anthropic_sdk", _raise_runtime_error)
    adapter = AnthropicAdapter(model="claude-x")

    with pytest.raises(AdapterError, match="construct") as excinfo:
        adapter.complete([{"role": "user", "content": "hi"}])

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_anthropic_wraps_default_client_constructor_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> Any:
        raise RuntimeError("missing api key")

    fake_sdk = SimpleNamespace(Anthropic=_boom)
    monkeypatch.setattr(anthropic_module, "_import_anthropic_sdk", lambda: fake_sdk)
    adapter = AnthropicAdapter(model="claude-x")

    with pytest.raises(AdapterError, match="construct") as excinfo:
        adapter.complete([{"role": "user", "content": "hi"}])

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "missing api key"


def test_anthropic_constructs_default_client_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = _RecordingEndpoint(_anthropic_response())
    fake_client = SimpleNamespace(messages=endpoint)
    fake_sdk = SimpleNamespace(Anthropic=lambda: fake_client)
    monkeypatch.setattr(anthropic_module, "_import_anthropic_sdk", lambda: fake_sdk)
    adapter = AnthropicAdapter(model="claude-x")

    response = adapter.complete([{"role": "user", "content": "hi"}])

    assert response.text == "hello"
    assert endpoint.calls[0]["model"] == "claude-x"


# --- openai adapter ------------------------------------------------------


def test_openai_maps_fields_exactly() -> None:
    endpoint = _RecordingEndpoint(_openai_response())
    adapter = OpenAIAdapter(
        model="gpt-x", client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    )

    response = adapter.complete([{"role": "user", "content": "hi"}], temperature=0.4, max_tokens=64)

    assert response == AdapterResponse(
        text="hello",
        reasoning=None,
        prompt_tokens=11,
        completion_tokens=7,
        stop_reason="stop",
        model="gpt-x",
        request_id="cmpl_1",
    )
    assert endpoint.calls == [
        {
            "model": "gpt-x",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.4,
            "max_completion_tokens": 64,
        }
    ]


def test_openai_omits_max_tokens_when_not_requested() -> None:
    endpoint = _RecordingEndpoint(_openai_response())
    adapter = OpenAIAdapter(
        model="gpt-x", client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    )

    adapter.complete([{"role": "user", "content": "hi"}])

    assert "max_completion_tokens" not in endpoint.calls[0]
    assert "max_tokens" not in endpoint.calls[0]


def test_openai_uses_first_choice_when_multiple_returned() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="first"), finish_reason="stop"),
            SimpleNamespace(message=SimpleNamespace(content="second"), finish_reason="length"),
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        model="gpt-x",
        id="cmpl_1",
    )
    endpoint = _RecordingEndpoint(response)
    adapter = OpenAIAdapter(
        model="gpt-x", client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    )

    result = adapter.complete([{"role": "user", "content": "hi"}])

    assert result.text == "first"
    assert result.stop_reason == "stop"


@pytest.mark.parametrize("finish_reason", [None, ""])
def test_openai_falls_back_to_unknown_stop_reason(finish_reason: str | None) -> None:
    endpoint = _RecordingEndpoint(_openai_response(finish_reason=finish_reason))
    adapter = OpenAIAdapter(
        model="gpt-x", client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    )

    assert adapter.complete([]).stop_reason == "unknown"


@pytest.mark.parametrize(
    "broken_response",
    [
        SimpleNamespace(),  # choices missing
        SimpleNamespace(choices=[]),  # choices empty
        SimpleNamespace(choices=[SimpleNamespace()]),  # message missing
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace())]),  # content missing
        SimpleNamespace(  # content not a str
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        ),
        SimpleNamespace(  # usage missing
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
        ),
        SimpleNamespace(  # prompt_tokens missing
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            usage=SimpleNamespace(completion_tokens=7),
        ),
        SimpleNamespace(  # negative token count
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            usage=SimpleNamespace(prompt_tokens=-1, completion_tokens=7),
        ),
        SimpleNamespace(  # float token count
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            usage=SimpleNamespace(prompt_tokens=1.5, completion_tokens=7),
        ),
    ],
)
def test_openai_malformed_shapes_raise_adapter_error(broken_response: Any) -> None:
    endpoint = _RecordingEndpoint(broken_response)
    adapter = OpenAIAdapter(
        model="gpt-x", client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    )

    with pytest.raises(AdapterError, match="malformed"):
        adapter.complete([{"role": "user", "content": "hi"}])


def test_openai_wraps_client_failures_preserving_cause() -> None:
    adapter = OpenAIAdapter(
        model="gpt-x",
        client=SimpleNamespace(chat=SimpleNamespace(completions=_ExplodingEndpoint())),
    )

    with pytest.raises(AdapterError, match="openai") as excinfo:
        adapter.complete([{"role": "user", "content": "hi"}])

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "transport exploded"


def test_openai_without_client_names_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error() -> Any:
        raise ImportError("simulated missing openai sdk")

    monkeypatch.setattr(openai_module, "_import_openai_sdk", _raise_import_error)
    adapter = OpenAIAdapter(model="gpt-x")

    with pytest.raises(AdapterError, match=r"awarebench\[openai\]") as excinfo:
        adapter.complete([{"role": "user", "content": "hi"}])

    assert isinstance(excinfo.value.__cause__, ImportError)


def test_openai_wraps_non_import_error_from_lazy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_runtime_error() -> Any:
        raise RuntimeError("sdk exploded on import")

    monkeypatch.setattr(openai_module, "_import_openai_sdk", _raise_runtime_error)
    adapter = OpenAIAdapter(model="gpt-x")

    with pytest.raises(AdapterError, match="construct") as excinfo:
        adapter.complete([{"role": "user", "content": "hi"}])

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_openai_wraps_default_client_constructor_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> Any:
        raise RuntimeError("missing api key")

    fake_sdk = SimpleNamespace(OpenAI=_boom)
    monkeypatch.setattr(openai_module, "_import_openai_sdk", lambda: fake_sdk)
    adapter = OpenAIAdapter(model="gpt-x")

    with pytest.raises(AdapterError, match="construct") as excinfo:
        adapter.complete([{"role": "user", "content": "hi"}])

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "missing api key"


def test_openai_constructs_default_client_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = _RecordingEndpoint(_openai_response())
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    fake_sdk = SimpleNamespace(OpenAI=lambda: fake_client)
    monkeypatch.setattr(openai_module, "_import_openai_sdk", lambda: fake_sdk)
    adapter = OpenAIAdapter(model="gpt-x")

    response = adapter.complete([{"role": "user", "content": "hi"}])

    assert response.text == "hello"
    assert endpoint.calls[0]["model"] == "gpt-x"


def test_openrouter_maps_fields_and_sends_exact_request() -> None:
    transport = _RecordingTransport(_openrouter_response())
    adapter = OpenRouterAdapter(
        model="vendor/model",
        api_key="test-secret",
        transport=transport,
    )

    response = adapter.complete(
        [{"role": "user", "content": "hi"}],
        temperature=0.4,
        max_tokens=64,
    )

    assert response == AdapterResponse(
        text="hello",
        reasoning="checked the trace",
        assistant_metadata={"reasoning": "checked the trace"},
        prompt_tokens=11,
        completion_tokens=7,
        stop_reason="stop",
        model="vendor/model",
        request_id="gen_1",
    )
    assert len(transport.calls) == 1
    request, timeout = transport.calls[0]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 120.0
    assert isinstance(request.data, bytes)
    assert json.loads(request.data.decode("utf-8")) == {
        "model": "vendor/model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.4,
        "max_tokens": 64,
    }


def test_openrouter_omits_max_tokens_when_not_requested() -> None:
    transport = _RecordingTransport(_openrouter_response())
    adapter = OpenRouterAdapter(model="vendor/model", api_key="key", transport=transport)

    adapter.complete([{"role": "user", "content": "hi"}])

    request, _ = transport.calls[0]
    assert isinstance(request.data, bytes)
    assert "max_tokens" not in json.loads(request.data.decode("utf-8"))


def test_openrouter_reads_api_key_lazily_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")
    transport = _RecordingTransport(_openrouter_response())
    adapter = OpenRouterAdapter(model="vendor/model", transport=transport)

    adapter.complete([])

    request, _ = transport.calls[0]
    assert request.get_header("Authorization") == "Bearer environment-key"


def test_openrouter_missing_api_key_raises_without_calling_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    transport = _RecordingTransport(_openrouter_response())
    adapter = OpenRouterAdapter(model="vendor/model", transport=transport)

    with pytest.raises(AdapterError, match="OPENROUTER_API_KEY"):
        adapter.complete([])

    assert transport.calls == []


def test_openrouter_wraps_transport_failure_without_exposing_key() -> None:
    error = HTTPError(
        "https://openrouter.ai/api/v1/chat/completions",
        429,
        "rate limited",
        Message(),
        BytesIO(b'{"error":"slow down"}'),
    )
    adapter = OpenRouterAdapter(
        model="vendor/model",
        api_key="do-not-print-this",
        transport=_ExplodingTransport(error),
    )

    with pytest.raises(AdapterError, match="429") as excinfo:
        adapter.complete([])

    assert excinfo.value.__cause__ is error
    assert "do-not-print-this" not in str(excinfo.value)


def test_openrouter_rejects_invalid_json() -> None:
    adapter = OpenRouterAdapter(
        model="vendor/model",
        api_key="key",
        transport=_RecordingTransport(b"not json"),
    )

    with pytest.raises(AdapterError, match="JSON"):
        adapter.complete([])


@pytest.mark.parametrize(
    "broken_response",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "hello"}}]},
        {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"completion_tokens": 7},
        },
        {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": True, "completion_tokens": 7},
        },
        {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": -1},
        },
        {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1.5},
        },
    ],
)
def test_openrouter_malformed_shapes_raise_adapter_error(
    broken_response: dict[str, Any],
) -> None:
    adapter = OpenRouterAdapter(
        model="vendor/model",
        api_key="key",
        transport=_RecordingTransport(broken_response),
    )

    with pytest.raises(AdapterError, match="malformed"):
        adapter.complete([])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response.pop("model"),
        lambda response: response.pop("id"),
        lambda response: response.__setitem__("object", "list"),
        lambda response: response["choices"][0].__setitem__("index", 7),
        lambda response: response["choices"][0].__setitem__("index", False),
        lambda response: response["choices"][0].__setitem__("finish_reason", None),
        lambda response: response["choices"][0]["message"].__setitem__("role", "user"),
        lambda response: response["usage"].__setitem__("total_tokens", 999),
    ],
)
def test_openrouter_rejects_missing_or_inconsistent_provenance(mutate: Any) -> None:
    response = _openrouter_response(reasoning=None)
    mutate(response)
    adapter = OpenRouterAdapter(
        model="vendor/model",
        api_key="key",
        transport=_RecordingTransport(response),
    )

    with pytest.raises(AdapterError, match="malformed"):
        adapter.complete([])


def test_openrouter_normalizes_structured_reasoning_text() -> None:
    response = _openrouter_response(reasoning=None)
    response["choices"][0]["message"]["reasoning_details"] = [
        {"type": "reasoning.summary", "summary": "summary observation"},
        {"type": "reasoning.text", "text": "first observation"},
        {"type": "reasoning.text", "text": "second observation"},
    ]
    adapter = OpenRouterAdapter(
        model="vendor/model",
        api_key="key",
        transport=_RecordingTransport(response),
    )

    result = adapter.complete([])

    assert result.reasoning == "summary observation\nfirst observation\nsecond observation"
    assert result.assistant_metadata == {
        "reasoning_details": response["choices"][0]["message"]["reasoning_details"]
    }


def test_openrouter_accepts_documented_reasoning_content_alias() -> None:
    response = _openrouter_response(reasoning=None)
    response["choices"][0]["message"]["reasoning_content"] = "alias observation"
    adapter = OpenRouterAdapter(
        model="vendor/model",
        api_key="key",
        transport=_RecordingTransport(response),
    )

    result = adapter.complete([])

    assert result.reasoning == "alias observation"
    assert result.assistant_metadata == {"reasoning_content": "alias observation"}


def test_openrouter_replays_nontext_and_nullable_reasoning_details_losslessly() -> None:
    response = _openrouter_response(reasoning=None)
    details = [
        {"type": "reasoning.text", "text": None, "id": "empty"},
        {"type": "reasoning.encrypted", "data": "ciphertext", "id": "encrypted"},
        {
            "type": "reasoning.server_tool_call",
            "name": "search",
            "arguments": {"query": "status"},
        },
    ]
    response["choices"][0]["message"]["reasoning_details"] = details
    adapter = OpenRouterAdapter(
        model="vendor/model",
        api_key="key",
        transport=_RecordingTransport(response),
    )

    result = adapter.complete([])

    assert result.reasoning is None
    assert result.assistant_metadata == {"reasoning_details": details}


def test_openrouter_default_transport_uses_redirect_rejecting_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, limit: int = -1) -> bytes:
            assert limit > 0
            return json.dumps(_openrouter_response()).encode("utf-8")

    calls: list[tuple[Request, float]] = []
    handlers: list[object] = []

    class _Opener:
        def open(self, request: Request, *, timeout: float) -> _Response:
            calls.append((request, timeout))
            return _Response()

    def _fake_build_opener(*items: object) -> _Opener:
        handlers.extend(items)
        return _Opener()

    monkeypatch.setattr(openrouter_module, "build_opener", _fake_build_opener)
    adapter = OpenRouterAdapter(model="vendor/model", api_key="key")

    result = adapter.complete([])

    assert result.text == "hello"
    assert len(calls) == 1
    assert calls[0][1] == 120.0
    assert any(isinstance(handler, openrouter_module._RejectRedirects) for handler in handlers)


def test_openrouter_redirect_handler_never_constructs_followup_request() -> None:
    handler = openrouter_module._RejectRedirects()
    original = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": "Bearer test-secret"},
    )

    redirected = handler.redirect_request(
        original,
        None,
        302,
        "Found",
        Message(),
        "https://evil.example/capture",
    )

    assert redirected is None


def test_mappers_tolerate_missing_model_and_request_id() -> None:
    bare_anthropic = SimpleNamespace(
        content=[SimpleNamespace(text="hello")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    bare_openai = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    anthropic_adapter = AnthropicAdapter(
        model="m", client=SimpleNamespace(messages=_RecordingEndpoint(bare_anthropic))
    )
    openai_adapter = OpenAIAdapter(
        model="m",
        client=SimpleNamespace(chat=SimpleNamespace(completions=_RecordingEndpoint(bare_openai))),
    )

    anthropic_result = anthropic_adapter.complete([])
    openai_result = openai_adapter.complete([])

    assert anthropic_result.model is None
    assert anthropic_result.request_id is None
    assert openai_result.model is None
    assert openai_result.request_id is None
