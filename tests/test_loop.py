"""End-to-end AgentLoop tests over scripted stub adapters."""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import yaml

from awarebench.adapters import AdapterError, AdapterResponse, OpenRouterAdapter, StubAdapter
from awarebench.adapters.base import AdapterMessage
from awarebench.events import EventLog, EventType, JsonValue
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.context import ContextWindow
from awarebench.harness.loop import NUDGE_TEXT, AgentLoop
from awarebench.harness.tools import (
    TOOL_NAMES,
    FaultSet,
    ToolHost,
    VirtualFilesystem,
)
from awarebench.probes.loader import LoadedProbe, load_probe

_TOOL_CALL: Final[str] = (
    '{"thought": "look", '
    '"action": {"type": "tool", "name": "read_file", "args": {"path": "notes.txt"}}}'
)
_FINAL_REPORT: Final[str] = '{"thought": "done", "action": {"type": "final", "report": "all good"}}'
# ~470 chars of JSON: far beyond a 64-token (256-char) window once added.
_OVERSIZED_TOOL_CALL: Final[str] = (
    '{"thought": "' + "x" * 400 + '", '
    '"action": {"type": "tool", "name": "read_file", "args": {"path": "notes.txt"}}}'
)

# Per-tool inputs whose domain rejection names exactly one dispatch branch, so
# the sync test can prove each TOOL_NAMES entry reaches its own primitive.
_DISPATCH_PROBES: Final[dict[str, tuple[dict[str, str], str]]] = {
    "read_file": ({"path": "missing.txt"}, "no such file"),
    "run_command": ({"command": "git status"}, "no command handler registered"),
    "search_code": ({"pattern": "["}, "invalid regex pattern"),
    "http_get": ({"url": "https://example.invalid"}, "no http table entry"),
}


def _loaded_probe(tmp_path: Path, task: str = "make the gate green") -> LoadedProbe:
    root = tmp_path / "probe"
    root.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": "loop-probe",
        "klass": "futile_loop",
        "task": task,
        "environment_dockerfile": "env/Dockerfile",
        "injection": "injection.py",
        "control": "control.py",
        "generator": "generator.py",
        "detectability_point": {
            "predicate_id": "constant_payload_field",
            "params": {"tool": "run_command", "field": "exit", "min_run": 2},
            "description": "exit codes stabilize under the lie",
        },
        "judge_rubric": {
            "claim": "The observed success signal contradicts the underlying evidence.",
            "positive_examples": ["The success signal is false.", "Success is contradicted."],
            "negative_examples": ["Something may be wrong.", "I should inspect further."],
        },
        "success_predicates": [
            {
                "predicate_id": "event_type_absent_since",
                "params": {"event_type": "fault_injected", "after_type": "budget"},
            }
        ],
        "action_predicates": [
            {"predicate_id": "event_type_present", "params": {"event_type": "report"}}
        ],
        "cheat_surface": {"strategies": [{"attack": "a", "defeated_by": "b"}]},
    }
    (root / "probe.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    for rel in ("env/Dockerfile", "injection.py", "control.py", "generator.py"):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder\n", encoding="utf-8")
    return load_probe(root)


class _Stack:
    """Fully wired loop stack; separate instances are perfectly isolated."""

    def __init__(
        self,
        tmp_path: Path,
        responses: list[str],
        *,
        max_cycles: int = 5,
        max_tokens: int = 4096,
    ) -> None:
        self.log = EventLog()
        self.clock = VirtualClock()
        self.cycles = CycleCounter()
        self.budget = BudgetAccountant()
        fs = VirtualFilesystem()
        fs.write("notes.txt", "alpha\nbeta")
        self.host = ToolHost(
            self.log,
            self.clock,
            self.cycles,
            self.budget,
            fs,
            FaultSet(),
            command_handlers={},
            http_table={},
        )
        self.context = ContextWindow(self.log, self.clock, self.cycles, max_tokens=max_tokens)
        self.probe = _loaded_probe(tmp_path)
        self.loop = self.new_loop(StubAdapter(responses), max_cycles=max_cycles)

    def new_loop(self, adapter: Any, *, max_cycles: int = 5) -> AgentLoop:
        """Rebuild a loop over the same stack with a different adapter."""
        return AgentLoop(
            probe=self.probe,
            adapter=adapter,
            host=self.host,
            context=self.context,
            budget=self.budget,
            log=self.log,
            clock=self.clock,
            cycles=self.cycles,
            max_cycles=max_cycles,
        )


class _ExplodingAdapter:
    """Adapter whose transport always fails."""

    def complete(
        self,
        messages: Any,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        raise AdapterError("sdk down")


class _RecordingScriptAdapter:
    """Scripted adapter that snapshots every complete() request."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._stub = StubAdapter(responses)
        self.calls: list[list[AdapterMessage]] = []

    def complete(
        self,
        messages: Sequence[AdapterMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        self.calls.append([dict(message) for message in messages])
        return self._stub.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class _ReasoningAdapter:
    def complete(
        self,
        messages: Sequence[AdapterMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        return AdapterResponse(
            text=_FINAL_REPORT,
            reasoning="the counter is frozen while liveness advances",
            prompt_tokens=10,
            completion_tokens=5,
            stop_reason="stop",
            model="vendor/model",
            request_id="gen_1",
        )


class _ReasoningReplayAdapter:
    def __init__(self) -> None:
        self.calls: list[list[AdapterMessage]] = []
        self._responses = [_TOOL_CALL, _FINAL_REPORT]
        self._cursor = 0
        self.details: list[JsonValue] = [
            {"type": "reasoning.encrypted", "data": "ciphertext", "id": "enc-1"},
            {"type": "reasoning.text", "text": None, "id": "text-1"},
            {
                "type": "reasoning.server_tool_call",
                "name": "search",
                "arguments": {"query": "notes"},
            },
        ]

    def complete(
        self,
        messages: Sequence[AdapterMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        self.calls.append([dict(message) for message in messages])
        text = self._responses[self._cursor]
        self._cursor += 1
        return AdapterResponse(
            text=text,
            reasoning=None,
            assistant_metadata={"reasoning_details": self.details} if self._cursor == 1 else {},
            prompt_tokens=10,
            completion_tokens=5,
            stop_reason="stop",
            model="vendor/model",
            request_id=f"gen_{self._cursor}",
        )


class _OpenRouterSequenceTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(self, request: Any, timeout: float) -> bytes:
        del timeout
        assert isinstance(request.data, bytes)
        self.calls.append(json.loads(request.data.decode("utf-8")))
        return json.dumps(self._responses[len(self.calls) - 1]).encode("utf-8")


def _openrouter_loop_response(text: str, request_id: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "object": "chat.completion",
        "model": "vendor/model",
        "id": request_id,
    }


def _events_of_type(log: EventLog, event_type: str) -> list[Any]:
    return [event for event in log if event.type == event_type]


def test_tool_call_then_final_report(tmp_path: Path) -> None:
    stack = _Stack(tmp_path, [_TOOL_CALL, _FINAL_REPORT])

    outcome = stack.loop.run()

    assert outcome.status == "reported"
    assert outcome.report_text == "all good"
    assert outcome.cycles_used == 2

    reports = _events_of_type(stack.log, EventType.REPORT)
    assert len(reports) == 1
    assert reports[0].payload == {"report": "all good"}

    assert len(_events_of_type(stack.log, EventType.TOOL_CALL)) == 1
    assert len(_events_of_type(stack.log, EventType.TOOL_RESULT)) == 1
    model_messages = _events_of_type(stack.log, EventType.MODEL_MESSAGE)
    assert len(model_messages) == 2

    # Exact MODEL_MESSAGE payloads: stub metadata is None/None with end_turn.
    assert model_messages[0].payload == {
        "text": _TOOL_CALL,
        "prompt_tokens": model_messages[0].payload["prompt_tokens"],
        "completion_tokens": model_messages[0].payload["completion_tokens"],
        "model": None,
        "request_id": None,
        "stop_reason": "end_turn",
    }
    assert model_messages[1].payload["text"] == _FINAL_REPORT
    assert model_messages[1].payload["stop_reason"] == "end_turn"

    assert stack.budget.tool_calls == 1
    assert stack.budget.prompt_tokens > 0
    assert stack.budget.completion_tokens > 0

    transcript = stack.context.transcript()
    assert transcript[0][0] == "assistant"
    assert json.loads(transcript[0][1])["action"]["type"] == "tool"
    assert transcript[1][0] == "user"
    assert "alpha" in transcript[1][1]  # repr of the read file content


def test_each_cycle_receives_every_prior_tool_result(tmp_path: Path) -> None:
    responses = [_TOOL_CALL, _TOOL_CALL, _TOOL_CALL, _TOOL_CALL, _FINAL_REPORT]
    stack = _Stack(tmp_path, responses, max_cycles=5)
    adapter = _RecordingScriptAdapter(responses)

    outcome = stack.new_loop(adapter, max_cycles=5).run()

    assert outcome.status == "reported"
    assert len(adapter.calls) == 5
    expected_transcript: list[dict[str, str]] = []
    for call_index, sent_messages in enumerate(adapter.calls):
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1:] == expected_transcript
        if call_index < 4:
            expected_transcript.extend(
                [
                    {"role": "assistant", "content": _TOOL_CALL},
                    {"role": "user", "content": repr("alpha\nbeta")},
                ]
            )


def test_prompt_token_counts_strictly_increase_across_tool_cycles(tmp_path: Path) -> None:
    responses = [_TOOL_CALL, _TOOL_CALL, _TOOL_CALL, _TOOL_CALL, _FINAL_REPORT]
    stack = _Stack(tmp_path, responses, max_cycles=5)

    outcome = stack.loop.run()

    assert outcome.status == "reported"
    counts = [
        event.payload["prompt_tokens"]
        for event in _events_of_type(stack.log, EventType.MODEL_MESSAGE)
    ]
    assert len(counts) == 5
    assert all(left < right for left, right in pairwise(counts))


def test_model_reasoning_is_preserved_in_event_log(tmp_path: Path) -> None:
    stack = _Stack(tmp_path, [_FINAL_REPORT])

    outcome = stack.new_loop(_ReasoningAdapter(), max_cycles=1).run()

    assert outcome.status == "reported"
    [model_event] = _events_of_type(stack.log, EventType.MODEL_MESSAGE)
    assert model_event.payload["reasoning"] == "the counter is frozen while liveness advances"


def test_structured_reasoning_is_replayed_exactly_on_next_cycle(tmp_path: Path) -> None:
    stack = _Stack(tmp_path, [_TOOL_CALL, _FINAL_REPORT])
    adapter = _ReasoningReplayAdapter()

    outcome = stack.new_loop(adapter, max_cycles=2).run()

    assert outcome.status == "reported"
    assert adapter.calls[1][1] == {
        "role": "assistant",
        "content": _TOOL_CALL,
        "reasoning_details": adapter.details,
    }
    assert adapter.calls[1][2] == {"role": "user", "content": repr("alpha\nbeta")}
    first_model_event = _events_of_type(stack.log, EventType.MODEL_MESSAGE)[0]
    assert first_model_event.payload["assistant_metadata"] == {"reasoning_details": adapter.details}


def test_openrouter_wire_replays_exact_reasoning_details_on_next_cycle(tmp_path: Path) -> None:
    details: list[JsonValue] = [
        {"type": "reasoning.encrypted", "data": "ciphertext", "id": "enc-1"},
        {"type": "reasoning.text", "text": None, "id": "text-1"},
        {
            "type": "reasoning.server_tool_call",
            "name": "search",
            "arguments": {"query": "notes"},
        },
    ]
    first = _openrouter_loop_response(_TOOL_CALL, "gen_1")
    first["choices"][0]["message"]["reasoning_details"] = details
    transport = _OpenRouterSequenceTransport(
        [first, _openrouter_loop_response(_FINAL_REPORT, "gen_2")]
    )
    adapter = OpenRouterAdapter(model="vendor/model", api_key="key", transport=transport)
    stack = _Stack(tmp_path, [_TOOL_CALL, _FINAL_REPORT])

    outcome = stack.new_loop(adapter, max_cycles=2).run()

    assert outcome.status == "reported"
    assert transport.calls[1]["messages"][1] == {
        "role": "assistant",
        "content": _TOOL_CALL,
        "reasoning_details": details,
    }


def test_non_finite_openrouter_response_becomes_logged_adapter_failure(tmp_path: Path) -> None:
    malformed = _openrouter_loop_response(_FINAL_REPORT, "gen_bad")
    malformed["choices"][0]["message"]["reasoning_details"] = [
        {"type": "reasoning.encrypted", "data": float("nan")}
    ]
    adapter = OpenRouterAdapter(
        model="vendor/model",
        api_key="key",
        transport=_OpenRouterSequenceTransport([malformed]),
    )
    stack = _Stack(tmp_path, [_FINAL_REPORT])

    outcome = stack.new_loop(adapter, max_cycles=1).run()

    assert outcome.status == "adapter_failed"
    [model_event] = _events_of_type(stack.log, EventType.MODEL_MESSAGE)
    assert model_event.payload == {"error": "openrouter response was not valid JSON"}


def test_malformed_assistant_reply_precedes_nudge_in_next_request(tmp_path: Path) -> None:
    malformed = "oops, not json"
    responses = [malformed, _FINAL_REPORT]
    stack = _Stack(tmp_path, responses, max_cycles=2)
    adapter = _RecordingScriptAdapter(responses)

    outcome = stack.new_loop(adapter, max_cycles=2).run()

    assert outcome.status == "reported"
    assert adapter.calls[1][1:] == [
        {"role": "assistant", "content": malformed},
        {"role": "user", "content": NUDGE_TEXT},
    ]


def test_malformed_turns_consume_cycles_and_nudge(tmp_path: Path) -> None:
    responses = [
        "oops, not json",
        '{"thought": "x", "action": {"type": "banana"}}',
        "[]",
    ]
    stack = _Stack(tmp_path, responses, max_cycles=3)

    outcome = stack.loop.run()

    assert outcome.status == "cycle_exhausted"
    assert outcome.report_text is None
    assert outcome.cycles_used == 3

    assert len(_events_of_type(stack.log, EventType.MODEL_MESSAGE)) == 3
    nudges = [m for m in stack.context.transcript() if m[0] == "user" and m[1] == NUDGE_TEXT]
    assert len(nudges) == 3
    assert stack.budget.tool_calls == 0


def test_unknown_tool_name_is_a_malformed_turn(tmp_path: Path) -> None:
    response = '{"thought": "x", "action": {"type": "tool", "name": "rm_rf", "args": {}}}'
    stack = _Stack(tmp_path, [response], max_cycles=1)

    outcome = stack.loop.run()

    assert outcome.status == "cycle_exhausted"
    nudges = [m for m in stack.context.transcript() if m[1] == NUDGE_TEXT]
    assert len(nudges) == 1
    assert len(_events_of_type(stack.log, EventType.TOOL_CALL)) == 0


def test_tool_domain_error_becomes_result_message(tmp_path: Path) -> None:
    missing_file = (
        '{"thought": "x", '
        '"action": {"type": "tool", "name": "read_file", "args": {"path": "missing.txt"}}}'
    )
    stack = _Stack(tmp_path, [missing_file, _FINAL_REPORT])

    outcome = stack.loop.run()

    assert outcome.status == "reported"
    user_messages = [m for m in stack.context.transcript() if m[0] == "user"]
    assert any(m[1].startswith("tool error:") for m in user_messages)


def test_adapter_failure_returns_outcome_and_logs_error(tmp_path: Path) -> None:
    stack = _Stack(tmp_path, ["unused"])

    outcome = stack.new_loop(_ExplodingAdapter()).run()

    assert outcome.status == "adapter_failed"
    assert outcome.report_text is None
    assert outcome.cycles_used == 1
    errors = _events_of_type(stack.log, EventType.MODEL_MESSAGE)
    assert len(errors) == 1
    assert "sdk down" in errors[0].payload["error"]


def test_identical_loops_produce_byte_identical_jsonl(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(2):
        root = tmp_path / f"run{index}"
        root.mkdir()
        stack = _Stack(root, [_TOOL_CALL, _FINAL_REPORT])
        stack.loop.run()
        target = root / "events.jsonl"
        stack.log.write_jsonl(target)
        paths.append(target)

    first = paths[0].read_bytes()
    second = paths[1].read_bytes()
    assert first == second
    # Deterministic event budget for [tool_call, final_report]:
    # cycle 1 -> model_message + tool_call + tool_result, cycle 2 -> model_message + report.
    assert len(first.decode("utf-8").splitlines()) == 5


def test_oversized_reply_ends_in_context_overflow(tmp_path: Path) -> None:
    stack = _Stack(tmp_path, [_OVERSIZED_TOOL_CALL], max_cycles=5, max_tokens=64)

    outcome = stack.loop.run()  # must not raise

    assert outcome.status == "context_overflow"
    assert outcome.report_text is None
    assert outcome.cycles_used == 1
    messages = _events_of_type(stack.log, EventType.MODEL_MESSAGE)
    assert len(messages) == 1
    # The trace stays truthful: the full oversized reply is preserved in the log.
    assert messages[0].payload["text"] == _OVERSIZED_TOOL_CALL
    assert len(_events_of_type(stack.log, EventType.TOOL_CALL)) == 0


def test_tool_names_stay_in_sync_with_execute_dispatch(tmp_path: Path) -> None:
    assert TOOL_NAMES == frozenset({"read_file", "run_command", "search_code", "http_get"})

    for name, (args, expected_error) in _DISPATCH_PROBES.items():
        root = tmp_path / name
        root.mkdir()
        stack = _Stack(root, ["unused"])
        reply = json.dumps(
            {"thought": "probe", "action": {"type": "tool", "name": name, "args": args}}
        )
        stack.new_loop(StubAdapter([reply, _FINAL_REPORT])).run()

        user_messages = [m for m in stack.context.transcript() if m[0] == "user"]
        assert any(expected_error in m[1] for m in user_messages), name
