"""Tests for the adapter layer: stub, vendor wrappers, and the protocol."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import awarebench.adapters.anthropic as anthropic_module
import awarebench.adapters.openai as openai_module
from awarebench.adapters import (
    AdapterError,
    AdapterResponse,
    AnthropicAdapter,
    ModelAdapter,
    OpenAIAdapter,
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


def _anthropic_response(
    text: str = "hello",
    stop_reason: str | None = "end_turn",
    input_tokens: int = 11,
    output_tokens: int = 7,
) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


def _openai_response(
    text: str = "hello",
    finish_reason: str | None = "stop",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


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


# --- AdapterResponse validation -----------------------------------------


def test_adapter_response_accepts_zero_tokens_and_reasoning_default() -> None:
    response = AdapterResponse(text="ok", prompt_tokens=0, completion_tokens=0, stop_reason="stop")

    assert response.reasoning is None


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

    assert isinstance(stub, ModelAdapter)
    assert isinstance(anthropic_adapter, ModelAdapter)
    assert isinstance(openai_adapter, ModelAdapter)


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
    )
    assert endpoint.calls == [
        {
            "model": "gpt-x",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.4,
            "max_tokens": 64,
        }
    ]


def test_openai_omits_max_tokens_when_not_requested() -> None:
    endpoint = _RecordingEndpoint(_openai_response())
    adapter = OpenAIAdapter(
        model="gpt-x", client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    )

    adapter.complete([{"role": "user", "content": "hi"}])

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
