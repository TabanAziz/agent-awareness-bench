"""Agent loop: the deterministic cycle between context, adapter, and tools.

Wire format (the model-response contract): each turn the model returns ONE
JSON object, either

    {"thought": "<reasoning>",
     "action": {"type": "tool", "name": "<tool>", "args": {...}}}

or

    {"thought": "<reasoning>", "action": {"type": "final", "report": "<text>"}}

"thought" is free-form reasoning and is not validated. Everything else about
the turn is validated strictly: the whole reply text must json.loads (after
whitespace stripping only - no code-fence tolerance), the action type must be
"tool" or "final", tool actions must name one of the four ToolHost primitives
and carry an args dict, final actions must carry a report string. Any
violation is a malformed turn, not an error: the raw reply is logged as a
MODEL_MESSAGE event, a deterministic nudge message enters the context, the
cycle is consumed, and the loop continues. Tool domain failures (missing
file, bad regex, ...) are equally non-fatal: the error repr becomes the
tool-result user message and the run continues.

The system message sent to the adapter is the probe task followed by these
protocol instructions; the rest of the request is the context transcript.
Every completion requests at most max_completion_tokens output tokens; the
Anthropic Messages API requires this parameter on every call, so the loop
always supplies it rather than leaving it to each adapter. Budget exhaustion
enforcement lands with cost-capped probes; the "budget_exhausted" outcome is
reserved until then.

When adding a turn to the context would overflow the window and no compaction
can make room, ContextWindow rejects the message. The loop never clamps or
truncates: it terminates with the truthful "context_overflow" outcome, and the
full reply text of the offending turn remains available in its MODEL_MESSAGE
event.
"""

from __future__ import annotations

import json
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from awarebench.adapters.base import AdapterError, AdapterMessage, ModelAdapter, complete_model
from awarebench.events import EventLog, EventType
from awarebench.harness._validation import require_strict_positive_int
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.context import ContextWindow
from awarebench.harness.tools import TOOL_NAMES, ToolHost
from awarebench.probes.loader import LoadedProbe

DEFAULT_CYCLE_STEP_US: Final[int] = 60_000_000
DEFAULT_MAX_COMPLETION_TOKENS: Final[int] = 2048

NUDGE_TEXT: Final[str] = (
    "your last message was not valid JSON per the protocol; respond with one JSON object"
)

_PROTOCOL_INSTRUCTIONS: Final[str] = (
    "You act in cycles. Every reply must be exactly one JSON object and "
    "nothing else. Either call a tool:\n"
    '{"thought": "<reasoning>", "action": {"type": "tool", "name": "<tool>", '
    '"args": {...}}}\n'
    "with <tool> one of read_file, run_command, search_code, http_get;\n"
    "or finish:\n"
    '{"thought": "<reasoning>", "action": {"type": "final", "report": "<text>"}}.\n'
    "No prose outside the JSON object."
)


class LoopOutcome(BaseModel):
    """Terminal state of one AgentLoop.run()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal[
        "reported",
        "cycle_exhausted",
        "budget_exhausted",
        "adapter_failed",
        "context_overflow",
    ]
    report_text: str | None
    cycles_used: int


class AgentLoop:
    """Runs the model/tool cycle until a final report or cycle exhaustion.

    Seeded from probe.manifest.generator_seed for future randomized
    scenarios; current behavior is fully deterministic with no randomness.
    """

    def __init__(
        self,
        *,
        probe: LoadedProbe,
        adapter: ModelAdapter,
        host: ToolHost,
        context: ContextWindow,
        budget: BudgetAccountant,
        log: EventLog,
        clock: VirtualClock,
        cycles: CycleCounter,
        max_cycles: int,
        cycle_step_us: int = DEFAULT_CYCLE_STEP_US,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        seed: int | None = None,
    ) -> None:
        require_strict_positive_int("max_cycles", max_cycles)
        require_strict_positive_int("cycle_step_us", cycle_step_us)
        require_strict_positive_int("max_completion_tokens", max_completion_tokens)
        self._probe = probe
        # The caller's seed (e.g. the CLI --seed flag) wins over the manifest
        # default so generator and loop always agree on randomness provenance.
        self._seed = seed if seed is not None else probe.manifest.generator_seed
        self._adapter = adapter
        self._host = host
        self._context = context
        self._budget = budget
        self._log = log
        self._clock = clock
        self._cycles = cycles
        self._max_cycles = max_cycles
        self._cycle_step_us = cycle_step_us
        self._max_completion_tokens = max_completion_tokens

    @property
    def seed(self) -> int:
        """Generator seed taken from the probe manifest."""
        return self._seed

    def run(self) -> LoopOutcome:
        """Consume up to max_cycles; returns the terminal LoopOutcome."""
        cycles_used = 0
        while cycles_used < self._max_cycles:
            self._cycles.advance()
            self._clock.advance_us(self._cycle_step_us)
            cycles_used += 1

            messages = self._build_messages()
            try:
                response = complete_model(
                    self._adapter,
                    messages,
                    temperature=0.0,
                    max_tokens=self._max_completion_tokens,
                )
            except AdapterError as exc:
                self._log.append(
                    EventType.MODEL_MESSAGE,
                    self._cycles.current,
                    self._clock.now_us,
                    {"error": str(exc)},
                )
                return LoopOutcome(
                    status="adapter_failed", report_text=None, cycles_used=cycles_used
                )

            model_payload: dict[str, Any] = {
                "text": response.text,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "model": response.model,
                "request_id": response.request_id,
                "stop_reason": response.stop_reason,
            }
            if response.reasoning is not None:
                model_payload["reasoning"] = response.reasoning
            if response.assistant_metadata:
                model_payload["assistant_metadata"] = response.assistant_metadata
            self._log.append(
                EventType.MODEL_MESSAGE,
                self._cycles.current,
                self._clock.now_us,
                model_payload,
            )
            self._budget.add_tokens(response.prompt_tokens, response.completion_tokens)

            action = _parse_action(response.text)
            if action is None:
                try:
                    self._context.add(
                        "assistant", response.text, metadata=response.assistant_metadata
                    )
                    self._context.add("user", NUDGE_TEXT)
                except ValueError:
                    return self._overflow(cycles_used)
                continue
            if action["type"] == "final":
                report_text = action["report"]
                self._log.append(
                    EventType.REPORT,
                    self._cycles.current,
                    self._clock.now_us,
                    {"report": report_text},
                )
                return LoopOutcome(
                    status="reported", report_text=report_text, cycles_used=cycles_used
                )

            try:
                self._context.add("assistant", response.text, metadata=response.assistant_metadata)
            except ValueError:
                return self._overflow(cycles_used)
            result_repr = execute_tool_action(self._host, action)
            try:
                self._context.add("user", result_repr)
            except ValueError:
                return self._overflow(cycles_used)

        return LoopOutcome(status="cycle_exhausted", report_text=None, cycles_used=cycles_used)

    def _overflow(self, cycles_used: int) -> LoopOutcome:
        """Terminal outcome when the window cannot hold the next transcript message."""
        return LoopOutcome(status="context_overflow", report_text=None, cycles_used=cycles_used)

    def _build_messages(self) -> list[AdapterMessage]:
        """System message (task + protocol) followed by the context transcript."""
        system_content = f"{self._probe.manifest.task}\n\n{_PROTOCOL_INSTRUCTIONS}"
        messages: list[AdapterMessage] = [{"role": "system", "content": system_content}]
        messages.extend(self._context.wire_transcript())
        return messages


def execute_tool_action(host: ToolHost, action: dict[str, Any]) -> str:
    """Dispatch one validated tool action and return the agent-visible result string."""
    name = action["name"]
    args = action["args"]
    calls_before = host.calls_seen
    try:
        if not isinstance(args, dict):
            raise ValueError("args must be an object")
        if name == "read_file":
            result: object = host.read_file(args["path"], args.get("start_line", 0))
        elif name == "run_command":
            result = host.run_command(args["command"])
        elif name == "search_code":
            result = host.search_code(args["pattern"])
        elif name == "http_get":
            result = host.http_get(args["url"])
        else:
            raise ValueError(f"unknown tool: {name}")
    except Exception as exc:  # noqa: BLE001 -- tool input must never crash a run
        if host.calls_seen == calls_before:
            return host.log_agent_tool_error(name, args, exc)
        return host.last_agent_visible_output()
    del result
    return host.last_agent_visible_output()


def _parse_action(text: str) -> dict[str, Any] | None:
    """Strictly parse the wire format; None marks a malformed turn."""
    try:
        parsed: object = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    action = parsed.get("action")
    if not isinstance(action, dict):
        return None
    kind = action.get("type")
    if kind == "final":
        if not isinstance(action.get("report"), str):
            return None
        return action
    if kind == "tool":
        name = action.get("name")
        if not isinstance(name, str) or name not in TOOL_NAMES:
            return None
        if not isinstance(action.get("args"), dict):
            return None
        return action
    return None
