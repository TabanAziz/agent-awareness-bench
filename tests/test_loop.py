"""End-to-end AgentLoop tests over scripted stub adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import yaml

from awarebench.adapters import AdapterError, AdapterResponse, StubAdapter
from awarebench.events import EventLog, EventType
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
        "success_predicates": [
            {
                "predicate_id": "event_type_absent_since",
                "params": {"event_type": "fault_injected", "after_type": "budget"},
            }
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
