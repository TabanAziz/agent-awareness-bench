"""Command-line entry point: run probes end to end.

Exit codes: 0 for a completed harness run, 2 when the probe loader rejects the
manifest (ProbeGateError), usage is invalid, or the target run already exists;
3 for unexpected exceptions (traceback on stderr); 4 for judge transport or
response failures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import shlex
import sys
import traceback
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Final

from awarebench.adapters import (
    AnthropicAdapter,
    ModelAdapter,
    OpenAIAdapter,
    OpenRouterAdapter,
    StubAdapter,
)
from awarebench.events import EventLog
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.context import ContextWindow, DropPolicy, drop_oldest, drop_oldest_half
from awarebench.harness.loop import AgentLoop
from awarebench.harness.stack import StackParts
from awarebench.harness.tools import ToolHost
from awarebench.probes.loader import LoadedProbe, ProbeGateError, load_probe
from awarebench.report import build_report
from awarebench.scoring.evaluate import evaluate
from awarebench.scoring.evaluate import passed as all_predicates_pass
from awarebench.scoring.judge import (
    JudgeError,
    NamedJudge,
    canonical_judge_model,
    judge_event_log,
)
from awarebench.scoring.judge_validation import (
    capture_validation_corpus,
    load_validation_labels,
    write_validation_corpus,
)
from awarebench.solvability import evaluate_cold_runs, trace_until_detectability

DEFAULT_CONTEXT_TOKENS: Final[int] = 16_384
MAX_SEED: Final[int] = (1 << 63) - 1

# Repeated forever when no stub script is given: every turn is malformed, so
# a scriptless stub run deterministically exhausts its cycles.
_MALFORMED_PLACEHOLDER: Final[str] = "not a valid protocol object"


def main(argv: list[str] | None = None) -> int:
    """Parse argv and execute the requested subcommand."""
    parser = argparse.ArgumentParser(prog="awarebench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one probe end to end.")
    run_parser.add_argument("probe_dir", type=Path)
    run_parser.add_argument(
        "--model",
        default="stub",
        help="stub, anthropic, openai, or openrouter:<model-id>.",
    )
    run_parser.add_argument("--model-name", default=None, help="Model id for vendor adapters.")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument(
        "--variant",
        choices=("fault", "control"),
        default="fault",
        help="Run the injected variant or its clean control.",
    )
    run_parser.add_argument("--max-cycles", type=int, default=40)
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum completion tokens requested per model call.",
    )
    run_parser.add_argument(
        "--context-tokens",
        type=int,
        default=DEFAULT_CONTEXT_TOKENS,
        help="Default context window size; a manifest context_max_tokens overrides it.",
    )
    run_parser.add_argument("--out", type=Path, default=Path("runs"))
    run_parser.add_argument(
        "--stub-script",
        type=Path,
        default=None,
        help="File with one JSON wire-format object per line (FIFO); "
        "without it the stub repeats a malformed placeholder and exhausts.",
    )

    judge_parser = subparsers.add_parser(
        "judge", help="Judge whether one completed run named the actual problem."
    )
    judge_parser.add_argument("probe_dir", type=Path)
    judge_parser.add_argument("--events", type=Path, required=True)
    judge_parser.add_argument(
        "--judge-model",
        action="append",
        default=[],
        help="Exactly two distinct vendor:model-id specifications.",
    )
    judge_parser.add_argument("--action-window-k", type=int, default=3)
    judge_parser.add_argument("--out", type=Path, required=True)

    capture_parser = subparsers.add_parser(
        "judge-validation-capture",
        help="Capture two isolated judge requests per frozen human-labelled excerpt.",
    )
    capture_parser.add_argument("labels", type=Path)
    capture_parser.add_argument("--probes-dir", type=Path, default=Path("probes"))
    capture_parser.add_argument(
        "--judge-model",
        action="append",
        default=[],
        help="Exactly two distinct vendor:model-id specifications.",
    )
    capture_parser.add_argument("--judged-at", type=date.fromisoformat, required=True)
    capture_parser.add_argument("--out", type=Path, required=True)

    solvability_parser = subparsers.add_parser(
        "solvability", help="Run ten cold solvability checks."
    )
    solvability_parser.add_argument("probe_dir", type=Path)
    solvability_parser.add_argument("--cold-model", required=True)
    solvability_parser.add_argument("--judge-model", action="append", required=True)
    solvability_parser.add_argument("--date", type=date.fromisoformat, required=True)
    solvability_parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "judge":
            return _judge_command(args)
        if args.command == "judge-validation-capture":
            return _judge_validation_capture_command(args)
        if args.command == "solvability":
            return _solvability_command(args)
        return _run_command(args)
    except ProbeGateError as exc:
        print(f"probe rejected: {exc}", file=sys.stderr)
        return 2
    except JudgeError as exc:
        print(f"judge failed: {exc}", file=sys.stderr)
        return 4
    except Exception:  # noqa: BLE001 -- CLI boundary turns crashes into exit code 3
        traceback.print_exc()
        return 3


_POLICY_FACTORIES: Final[dict[str, DropPolicy]] = {
    "drop_oldest": drop_oldest,
    "drop_oldest_half": drop_oldest_half,
}


def _policy_by_name(name: str | None) -> DropPolicy:
    """Resolve a variant-named drop policy; None and unknown fall back to oldest."""
    if name is None:
        return _POLICY_FACTORIES["drop_oldest"]
    return _POLICY_FACTORIES.get(name, _POLICY_FACTORIES["drop_oldest"])


def _load_artifact(path: Path, module_name: str) -> ModuleType:
    """Import a probe artifact module from its resolved path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load probe artifact: {path.name}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations via sys.modules[module.__name__];
    # registering also makes sibling imports between artifacts resolvable.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_stack(
    loaded: LoadedProbe,
    log: EventLog,
    clock: VirtualClock,
    cycles: CycleCounter,
    *,
    seed: int,
    variant: str,
) -> StackParts:
    """Derive the seeded variant and let the probe's artifacts fill the stack."""
    stem = f"awb_probe_{loaded.manifest.id}"
    generator = _load_artifact(loaded.generator, f"{stem}_generator")
    # Injection/control modules import their sibling generator by name; make
    # the just-loaded module resolvable so each probe sees its own copy.
    sys.modules["generator"] = generator
    parts = StackParts(clock=clock, cycles=cycles, log=log)
    artifact_path = loaded.injection if variant == "fault" else loaded.control
    applier = _load_artifact(artifact_path, f"{stem}_{variant}")
    applier.apply(parts, seed, log, generator.generate(seed))
    return parts


def _run_command(args: argparse.Namespace) -> int:
    try:
        _validate_run_numbers(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        backend, model_id = _parse_model_spec(args.model, args.model_name)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.stub_script is not None and not args.stub_script.is_file():
        print(f"--stub-script not found: {args.stub_script}", file=sys.stderr)
        return 2

    loaded = load_probe(args.probe_dir)

    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()
    parts = _build_stack(loaded, log, clock, cycles, seed=args.seed, variant=args.variant)
    host = ToolHost(
        log,
        clock,
        cycles,
        budget,
        parts.fs,
        parts.faults,
        command_handlers=parts.command_handlers,
        http_table=parts.http_table,
    )
    # Window sizing precedence: probe variant override, then the manifest,
    # then the flag. Policy comes from the variant when it names one.
    window_tokens = (
        parts.context_max_tokens
        if parts.context_max_tokens is not None
        else loaded.manifest.context_max_tokens
    )
    policy = _policy_by_name(parts.drop_policy)
    context = ContextWindow(
        log,
        clock,
        cycles,
        max_tokens=window_tokens if window_tokens is not None else args.context_tokens,
        policy=policy,
    )
    # Seeded messages enter the transcript before any loop cycle so they hold
    # the earliest seqs and compact away first under pressure.
    for role, content in parts.seed_messages:
        context.add(role, content)

    out_dir = args.out / loaded.manifest.id / _run_label(backend, model_id, args.variant, args.seed)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"run output already exists: {out_dir}", file=sys.stderr)
        return 2

    adapter = _build_adapter(args, backend=backend, model_id=model_id)
    outcome = AgentLoop(
        probe=loaded,
        adapter=adapter,
        host=host,
        context=context,
        budget=budget,
        log=log,
        clock=clock,
        cycles=cycles,
        max_cycles=args.max_cycles,
        max_completion_tokens=args.max_tokens,
        seed=args.seed,
    ).run()

    snapshot = budget.snapshot()
    score = evaluate(loaded, log, control=args.variant == "control")
    log.write_jsonl(out_dir / "events.jsonl")
    build_report(
        loaded,
        backend=backend,
        requested_model=model_id,
        variant=args.variant,
        seed=args.seed,
        outcome=outcome,
        budget_snapshot=snapshot,
        predicates=score,
    ).write_json(out_dir / "report.json")
    print(
        f"outcome={outcome.status} cycles={outcome.cycles_used} "
        f"tokens={snapshot['prompt_tokens']}+{snapshot['completion_tokens']} "
        f"passed={all_predicates_pass(score)}"
    )
    return 0


def _judge_command(args: argparse.Namespace) -> int:
    requested_models: list[str] = args.judge_model
    if len(requested_models) != 2:
        print("exactly two --judge-model values are required", file=sys.stderr)
        return 2
    try:
        models = [canonical_judge_model(model) for model in requested_models]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if models[0] == models[1]:
        print("--judge-model values must be distinct", file=sys.stderr)
        return 2
    if (
        isinstance(args.action_window_k, bool)
        or not isinstance(args.action_window_k, int)
        or args.action_window_k < 1
    ):
        print("--action-window-k must be >= 1", file=sys.stderr)
        return 2
    if not args.events.is_file():
        print(f"events file not found: {args.events}", file=sys.stderr)
        return 2
    if args.out.exists():
        print(f"judge output already exists: {args.out}", file=sys.stderr)
        return 2
    try:
        adapters = [_build_judge_adapter(model) for model in models]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    loaded = load_probe(args.probe_dir)
    log = EventLog.read_jsonl(args.events)
    result = judge_event_log(
        loaded,
        log,
        judges=(
            NamedJudge(models[0], adapters[0]),
            NamedJudge(models[1], adapters[1]),
        ),
        action_window_k=args.action_window_k,
    )
    result.write_json(args.out)
    detection = "unresolved" if result.detected is None else str(result.detected).lower()
    detection_latency = (
        "unavailable" if result.detection_latency is None else str(result.detection_latency)
    )
    action_gap = "unavailable" if result.action_gap is None else str(result.action_gap)
    print(
        f"detection={detection} detection_latency={detection_latency} action_gap={action_gap} "
        f"judge_disagreement_rate={result.disagreement_rate}"
    )
    return 0


def _solvability_command(args: argparse.Namespace) -> int:
    if args.out.exists():
        print(f"solvability output already exists: {args.out}", file=sys.stderr)
        return 2
    if len(args.judge_model) != 2:
        print("exactly two --judge-model values are required", file=sys.stderr)
        return 2
    try:
        cold_adapter = _build_judge_adapter(args.cold_model)
        judges = tuple(NamedJudge(model, _build_judge_adapter(model)) for model in args.judge_model)
        loaded = load_probe(args.probe_dir, require_solvability=False)
        result = evaluate_cold_runs(
            trace=lambda seed: trace_until_detectability(
                args.probe_dir,
                seed,
                stack_builder=lambda probe, log, clock, cycles, run_seed, variant: _build_stack(
                    probe, log, clock, cycles, seed=run_seed, variant=variant
                ),
                policy_by_name=_policy_by_name,
                default_context_tokens=DEFAULT_CONTEXT_TOKENS,
            ),
            rubric=loaded.manifest.judge_rubric,
            cold_model=args.cold_model,
            cold_adapter=cold_adapter,
            judges=judges,  # type: ignore[arg-type]
            today=args.date,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.out.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"solvability={result.passed_count}/{result.count}")
    return 0


def _build_judge_adapter(model_spec: str) -> ModelAdapter:
    backend, separator, model_id = model_spec.partition(":")
    if not separator or not model_id.strip():
        raise ValueError("judge models must use vendor:model-id syntax")
    model_id = model_id.strip()
    if backend == "anthropic":
        return AnthropicAdapter(model=model_id)
    if backend == "openai":
        return OpenAIAdapter(model=model_id)
    if backend == "openrouter":
        return OpenRouterAdapter(model=model_id)
    raise ValueError(f"unsupported judge model backend: {backend}")


def _judge_validation_capture_command(args: argparse.Namespace) -> int:
    requested_models: list[str] = args.judge_model
    if len(requested_models) != 2:
        print("exactly two --judge-model values are required", file=sys.stderr)
        return 2
    try:
        models = [canonical_judge_model(model) for model in requested_models]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if models[0] == models[1]:
        print("--judge-model values must be distinct", file=sys.stderr)
        return 2
    if not args.labels.is_file():
        print(f"validation labels file not found: {args.labels}", file=sys.stderr)
        return 2
    if args.out.exists():
        print(f"judge validation output already exists: {args.out}", file=sys.stderr)
        return 2
    probe_paths = sorted(args.probes_dir.glob("*/*/probe.yaml"))
    if not probe_paths:
        print(f"no probe manifests found under: {args.probes_dir}", file=sys.stderr)
        return 2
    try:
        adapters = [_build_judge_adapter(model) for model in models]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    loaded = [load_probe(path.parent) for path in probe_paths]
    rubrics = {probe.manifest.id: probe.manifest.judge_rubric for probe in loaded}
    labels = load_validation_labels(args.labels)
    capture_argv = [
        "awarebench",
        "judge-validation-capture",
        args.labels.as_posix(),
        "--probes-dir",
        args.probes_dir.as_posix(),
        "--judge-model",
        models[0],
        "--judge-model",
        models[1],
        "--judged-at",
        args.judged_at.isoformat(),
        "--out",
        args.out.as_posix(),
    ]
    cases = capture_validation_corpus(
        labels,
        rubrics,
        judges=(
            NamedJudge(models[0], adapters[0]),
            NamedJudge(models[1], adapters[1]),
        ),
        judged_at=args.judged_at,
        capture_command=shlex.join(capture_argv),
    )
    write_validation_corpus(cases, args.out)
    print(f"captured={len(cases)} judge_requests={len(cases) * 2}")
    return 0


def _parse_model_spec(model: str, model_name: str | None) -> tuple[str, str | None]:
    """Resolve one CLI model spec into its adapter backend and vendor model ID."""
    if model == "stub":
        return model, None
    if model in {"anthropic", "openai"}:
        if not model_name:
            raise ValueError(f"--model-name is required for --model {model}")
        return model, model_name
    if model.startswith("openrouter:"):
        openrouter_model = model.removeprefix("openrouter:").strip()
        if not openrouter_model:
            raise ValueError("openrouter model id is required after --model openrouter:")
        if model_name is not None:
            raise ValueError("--model-name cannot be combined with an openrouter:<id> spec")
        return "openrouter", openrouter_model
    raise ValueError(f"unsupported --model: {model}")


def _build_adapter(
    args: argparse.Namespace,
    *,
    backend: str,
    model_id: str | None,
) -> ModelAdapter:
    if backend == "stub":
        return StubAdapter(_read_stub_script(args.stub_script))
    if model_id is None:
        raise ValueError(f"model id missing for {backend}")
    if backend == "anthropic":
        return AnthropicAdapter(model=model_id)
    if backend == "openai":
        return OpenAIAdapter(model=model_id)
    if backend == "openrouter":
        return OpenRouterAdapter(model=model_id)
    raise ValueError(f"unsupported adapter backend: {backend}")


def _run_label(backend: str, model_id: str | None, variant: str, seed: int) -> str:
    """Return a bounded, collision-resistant path component for one run."""
    _require_seed(seed)
    if backend == "stub" and model_id is None:
        return f"stub-{variant}-s{seed}"
    model_spec = backend if model_id is None else f"{backend}:{model_id}"
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_spec)
    safe_model = re.sub(r"-+", "-", safe_model).strip("-.") or "model"
    prefix = safe_model[:48].rstrip("-.") or "model"
    digest = hashlib.sha256(model_spec.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}-{variant}-s{seed}"


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")


def _validate_run_numbers(args: argparse.Namespace) -> None:
    _require_seed(args.seed)
    for field, option in (
        ("max_cycles", "max-cycles"),
        ("max_tokens", "max-tokens"),
        ("context_tokens", "context-tokens"),
    ):
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{option} must be >= 1")


def _read_stub_script(path: Path | None) -> list[str]:
    """One JSON object per non-empty line; missing or empty file means placeholder."""
    if path is None:
        return [_MALFORMED_PLACEHOLDER]
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines if lines else [_MALFORMED_PLACEHOLDER]
