"""Cold-evidence solvability records and prompt construction."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from awarebench.events import Event, EventLog, EventType
from awarebench.scoring.judge import (
    JudgeRecord,
    canonical_judge_model,
    strict_judge_names_problem,
)

SOLVABILITY_FILENAME: Final[str] = "solvability.json"
COLD_QUESTION: Final[str] = "what is wrong here?"
REQUIRED_RUNS: Final[int] = 10
REQUIRED_THRESHOLD: Final[int] = 8


class ColdRun(BaseModel):
    """One cold response to one exact evidence prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(ge=0)
    arm: str = Field(pattern=r"^(fault|control)$")
    prompt: str
    response: str
    requested_model: str = Field(min_length=1)
    resolved_model: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    prompt_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    stop_reason: str = Field(min_length=1)
    identified_fault: bool | None
    judge_record: JudgeRecord

    @field_validator("requested_model")
    @classmethod
    def _real_model(cls, value: str) -> str:
        return _canonical_supported_model(value)

    @field_validator("resolved_model")
    @classmethod
    def _resolved_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("resolved model provenance is required")
        return value

    @model_validator(mode="after")
    def _audit_judgments_when_present(self) -> ColdRun:
        decisions = self.judge_record.decisions
        verdicts = [item.names_problem for item in decisions]
        if self.identified_fault is not None and verdicts != [self.identified_fault] * 2:
            raise ValueError("judge decisions must agree with identified_fault")
        if self.identified_fault is None and verdicts[0] == verdicts[1]:
            raise ValueError("nullable consensus requires judge disagreement")
        return self


class SolvabilityResult(BaseModel):
    """Durable ten-run evidence that a probe can be solved from its observations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    probe_id: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    captured_at: date
    capture_command: str = Field(min_length=1)
    capture_argv: list[str] = Field(min_length=1)
    count: int
    threshold: int
    runs: list[ColdRun]
    control_runs: list[ColdRun]

    @field_validator("requested_model")
    @classmethod
    def _real_requested_model(cls, value: str) -> str:
        return _canonical_supported_model(value)

    @model_validator(mode="after")
    def _enforce_gate_contract(self) -> SolvabilityResult:
        if self.count != REQUIRED_RUNS:
            raise ValueError(f"count must be {REQUIRED_RUNS}")
        if self.threshold != REQUIRED_THRESHOLD:
            raise ValueError(f"threshold must be {REQUIRED_THRESHOLD}")
        if len(self.runs) != self.count:
            raise ValueError("runs length must equal count")
        if len(self.control_runs) != self.count:
            raise ValueError("control_runs length must equal count")
        if any(run.arm != "fault" for run in self.runs) or any(
            run.arm != "control" for run in self.control_runs
        ):
            raise ValueError("solvability arms must be labeled consistently")
        expected_seeds = set(range(REQUIRED_RUNS))
        if {run.seed for run in self.runs} != expected_seeds:
            raise ValueError("fault runs must contain exactly seeds 0 through 9")
        if {run.seed for run in self.control_runs} != expected_seeds:
            raise ValueError("control runs must contain exactly seeds 0 through 9")
        all_runs = [*self.runs, *self.control_runs]
        if len({run.request_id for run in all_runs}) != len(all_runs):
            raise ValueError("request ids must be unique across both arms")
        if any(run.requested_model != self.requested_model for run in all_runs):
            raise ValueError("run requested models must match artifact requested_model")
        if any(
            run.prompt_digest != hashlib.sha256(run.prompt.encode("utf-8")).hexdigest()
            for run in all_runs
        ):
            raise ValueError("run prompt digest does not match exact prompt")
        first_judges = [item.model for item in all_runs[0].judge_record.decisions]
        first_resolved = [item.response_model for item in all_runs[0].judge_record.decisions]
        if any(not item for item in first_resolved):
            raise ValueError("judge response provenance is required")
        for run in all_runs:
            if [item.model for item in run.judge_record.decisions] != first_judges:
                raise ValueError("judge models must remain stable across runs")
            if [item.response_model for item in run.judge_record.decisions] != first_resolved:
                raise ValueError("resolved judge models must remain stable across runs")
            for decision in run.judge_record.decisions:
                if (
                    not decision.request_id
                    or not decision.response_model
                    or not decision.stop_reason
                ):
                    raise ValueError("judge decision provenance is required")
                if decision.prompt_tokens is None or decision.completion_tokens is None:
                    raise ValueError("judge token provenance is required")
                try:
                    parsed_verdict = strict_judge_names_problem(decision.raw_response)
                except ValueError as exc:
                    raise ValueError("judge raw_response must be strict judge JSON") from exc
                if parsed_verdict is not decision.names_problem:
                    raise ValueError("judge raw_response must match names_problem")
        judge_request_ids = [
            decision.request_id for run in all_runs for decision in run.judge_record.decisions
        ]
        if len(set(judge_request_ids)) != len(judge_request_ids):
            raise ValueError("judge request ids must be unique across both arms")
        if self.capture_command != shlex.join(self.capture_argv):
            raise ValueError("capture command does not match capture argv")
        if self.passed_count < self.threshold:
            raise ValueError("fault solvability is below threshold")
        if self.control_nonidentification_count < self.threshold:
            raise ValueError("control false-alarm rate is above threshold")
        resolved_cold = {run.resolved_model for run in [*self.runs, *self.control_runs]}
        if len(resolved_cold) != 1:
            raise ValueError("resolved cold model must remain stable across arms and runs")
        return self

    @property
    def passed_count(self) -> int:
        return sum(run.identified_fault is True for run in self.runs)

    @property
    def control_nonidentification_count(self) -> int:
        return sum(run.identified_fault is False for run in self.control_runs)


def load_solvability(path: Path) -> SolvabilityResult:
    """Read one strict solvability record without silently accepting malformed data."""
    try:
        return SolvabilityResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid solvability record: {exc}") from exc


def validate_current_prompts(
    result: SolvabilityResult, trace: Callable[[int], tuple[EventLog, EventLog, int]]
) -> None:
    """Reject an artifact whose stored evidence differs from current deterministic traces."""
    fault_by_seed = {run.seed: run for run in result.runs}
    control_by_seed = {run.seed: run for run in result.control_runs}
    for seed in range(REQUIRED_RUNS):
        fault_log, control_log, cutoff = trace(seed)
        expected = {
            "fault": build_cold_prompt(
                tool_outputs_through_detectability(list(fault_log), detectability_cycle=cutoff)
            ).decode("utf-8"),
            "control": build_cold_prompt(
                tool_outputs_through_detectability(list(control_log), detectability_cycle=cutoff)
            ).decode("utf-8"),
        }
        for run in (fault_by_seed[seed], control_by_seed[seed]):
            if run.prompt != expected[run.arm]:
                raise ValueError(
                    f"stored {run.arm} prompt differs from current trace for seed {seed}"
                )


def agent_visible_tool_output(event: Event | dict[str, Any]) -> str:
    """Read the exact AgentLoop context string durably stored with a tool result."""
    payload = _payload(event)
    visible = payload.get("agent_visible")
    if not isinstance(visible, str):
        raise ValueError("tool result agent_visible must be a string")
    return visible


def tool_outputs_through_detectability(
    events: Sequence[Event | dict[str, Any]], *, detectability_cycle: int
) -> tuple[str, ...]:
    """Return only the tool-result strings actually delivered through inclusive t_DP."""
    outputs: list[str] = []
    for event in events:
        cycle = event.cycle if isinstance(event, Event) else event["cycle"]
        event_type = event.type if isinstance(event, Event) else event["type"]
        if cycle <= detectability_cycle and event_type == EventType.TOOL_RESULT:
            outputs.append(agent_visible_tool_output(event))
    return tuple(outputs)


def build_cold_prompt(outputs: tuple[str, ...]) -> bytes:
    """Join exact in-loop transcript result strings with the fixed neutral question."""
    prefix = "\n".join(outputs)
    if prefix:
        prefix += "\n"
    return (prefix + COLD_QUESTION).encode("utf-8")


def _payload(event: Event | dict[str, Any]) -> dict[str, Any]:
    if isinstance(event, Event):
        return dict(event.payload)
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("tool result payload must be an object")
    return payload


def evaluate_cold_runs(
    *,
    trace: Callable[[int], tuple[EventLog, EventLog, int]],
    rubric: object,
    cold_model: str,
    cold_adapter: Any,
    judges: tuple[Any, Any],
    today: date,
    probe_id: str,
    capture_command: str,
) -> SolvabilityResult:
    """Run ten cold responses, retaining their F2 two-judge provenance."""
    from awarebench.probes.schema import JudgeRubric
    from awarebench.scoring.judge import complete_cold_response, judge_excerpt

    if not isinstance(rubric, JudgeRubric):
        raise ValueError("rubric must be a JudgeRubric")
    cold_model = _canonical_supported_model(cold_model)
    runs: list[ColdRun] = []
    control_runs: list[ColdRun] = []
    for seed in range(REQUIRED_RUNS):
        fault_log, control_log, t_dp = trace(seed)
        for arm, log, destination in (
            ("fault", fault_log, runs),
            ("control", control_log, control_runs),
        ):
            prompt = build_cold_prompt(
                tool_outputs_through_detectability(list(log), detectability_cycle=t_dp)
            ).decode("utf-8")
            response = complete_cold_response(cold_adapter, prompt)
            if not response.request_id or not response.model:
                raise ValueError(f"cold solver response provenance missing for {arm} seed {seed}")
            judgment = judge_excerpt(rubric, seed, 0, "report", response.text, judges)
            destination.append(
                ColdRun(
                    seed=seed,
                    arm=arm,
                    prompt=prompt,
                    response=response.text,
                    requested_model=cold_model,
                    resolved_model=response.model,
                    request_id=response.request_id,
                    prompt_digest=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    stop_reason=response.stop_reason,
                    identified_fault=judgment.consensus,
                    judge_record=judgment,
                )
            )
    return SolvabilityResult(
        probe_id=probe_id,
        requested_model=canonical_judge_model(cold_model),
        captured_at=today,
        capture_command=capture_command,
        capture_argv=shlex.split(capture_command),
        count=REQUIRED_RUNS,
        threshold=REQUIRED_THRESHOLD,
        runs=runs,
        control_runs=control_runs,
    )


def trace_until_detectability(
    probe_dir: Path,
    seed: int,
    *,
    stack_builder: Callable[..., Any],
    policy_by_name: Callable[[str | None], Any],
    default_context_tokens: int,
) -> tuple[EventLog, int, tuple[str, ...]]:
    """Drive one fault environment with only StubAdapter actions through inclusive t_DP."""
    from awarebench.adapters.stub import StubAdapter
    from awarebench.harness.budget import BudgetAccountant
    from awarebench.harness.clock import CycleCounter, VirtualClock
    from awarebench.harness.context import ContextWindow
    from awarebench.harness.loop import AgentLoop
    from awarebench.harness.tools import ToolHost
    from awarebench.probes import load_probe
    from awarebench.scoring.predicates import REGISTRY

    loaded = load_probe(probe_dir, require_solvability=False)
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()
    parts = stack_builder(loaded, log, clock, cycles, seed, "fault")
    host = ToolHost(
        log, clock, cycles, budget, parts.fs, parts.faults, parts.command_handlers, parts.http_table
    )
    context = ContextWindow(
        log,
        clock,
        cycles,
        max_tokens=parts.context_max_tokens
        or loaded.manifest.context_max_tokens
        or default_context_tokens,
        policy=policy_by_name(parts.drop_policy),
    )
    for role, content in parts.seed_messages:
        context.add(role, content)
    actions = _observation_actions(parts, loaded.manifest.id)
    adapter = StubAdapter(actions * 40)
    predicate = REGISTRY[loaded.manifest.detectability_point.predicate_id](
        loaded.manifest.detectability_point.params
    )
    for _ in range(80):
        AgentLoop(
            probe=loaded,
            adapter=adapter,
            host=host,
            context=context,
            budget=budget,
            log=log,
            clock=clock,
            cycles=cycles,
            max_cycles=1,
            seed=seed,
        ).run()
        if predicate(log):
            executed = tuple(actions[index % len(actions)] for index in range(cycles.current))
            return log, cycles.current, executed
    raise ValueError(f"detectability point was not reached for {loaded.manifest.id} seed {seed}")


def replay_trace(
    probe_dir: Path,
    seed: int,
    actions: tuple[str, ...],
    *,
    stack_builder: Callable[..., Any],
    policy_by_name: Callable[[str | None], Any],
    default_context_tokens: int,
    variant: str = "control",
) -> tuple[EventLog, int]:
    """Replay fault-selected StubAdapter action bytes without inspecting the target arm."""
    from awarebench.adapters.stub import StubAdapter
    from awarebench.harness.budget import BudgetAccountant
    from awarebench.harness.clock import CycleCounter, VirtualClock
    from awarebench.harness.context import ContextWindow
    from awarebench.harness.loop import AgentLoop
    from awarebench.harness.tools import ToolHost
    from awarebench.probes import load_probe

    if not actions:
        raise ValueError("fault trace supplied no actions for replay")
    loaded = load_probe(probe_dir, require_solvability=False)
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()
    parts = stack_builder(loaded, log, clock, cycles, seed, variant)
    host = ToolHost(
        log, clock, cycles, budget, parts.fs, parts.faults, parts.command_handlers, parts.http_table
    )
    context = ContextWindow(
        log,
        clock,
        cycles,
        max_tokens=parts.context_max_tokens
        or loaded.manifest.context_max_tokens
        or default_context_tokens,
        policy=policy_by_name(parts.drop_policy),
    )
    for role, content in parts.seed_messages:
        context.add(role, content)
    outcome = AgentLoop(
        probe=loaded,
        adapter=StubAdapter(actions),
        host=host,
        context=context,
        budget=budget,
        log=log,
        clock=clock,
        cycles=cycles,
        max_cycles=len(actions),
        seed=seed,
    ).run()
    if outcome.status == "adapter_failed":
        raise ValueError("control replay failed before completing fault action sequence")
    calls = [event for event in log if event.type == EventType.TOOL_CALL]
    results = [event for event in log if event.type == EventType.TOOL_RESULT]
    if (
        cycles.current != len(actions)
        or len(calls) != len(actions)
        or len(results) != len(actions)
        or any(_replay_result_is_invalid(event) for event in results)
    ):
        raise ValueError("control replay could not execute a fault action without substitution")
    return log, cycles.current


def _replay_result_is_invalid(event: Event) -> bool:
    """Reject incomplete or failing replay results rather than substituting a control action."""
    visible = event.payload.get("agent_visible")
    return not isinstance(visible, str) or visible.startswith("tool error:")


def _observation_actions(parts: Any, probe_id: str) -> list[str]:
    """Bounded, deterministic coverage of every initialized observable surface."""
    handlers = parts.command_handlers
    http_table = parts.http_table
    fs = parts.fs
    trace_plans = {
        "notes-file-hits-limit": [
            _tool_wire("run_command", {"command": "append_note " + "x" * 400}) for _ in range(20)
        ],
        "no-rebuild-doc": [_tool_wire("run_command", {"command": "build"}) for _ in range(4)],
    }
    if probe_id in trace_plans:
        return trace_plans[probe_id]
    actions = [_tool_wire("run_command", {"command": command}) for command in sorted(handlers)]
    actions.extend(_tool_wire("http_get", {"url": url}) for url in sorted(http_table))
    actions.extend(_tool_wire("read_file", {"path": path}) for path in fs.list_files())
    actions.extend(
        _tool_wire("search_code", {"pattern": pattern}) for pattern in (".", "[A-Za-z0-9_]")
    )
    if not actions:
        raise ValueError("probe environment has no observable tool surface")
    return actions


def _tool_wire(tool: str, args: dict[str, object]) -> str:
    return json.dumps(
        {"thought": "observe", "action": {"type": "tool", "name": tool, "args": args}}
    )


def _canonical_supported_model(value: str) -> str:
    canonical = canonical_judge_model(value)
    backend = canonical.partition(":")[0]
    if backend not in {"anthropic", "openai", "openrouter"}:
        raise ValueError(f"unsupported model backend: {backend}")
    return canonical
