"""Command-line entry point: run probes end to end.

Exit codes: 0 for any completed run regardless of probe outcome, 2 when the
probe loader rejects the manifest (ProbeGateError) or usage is invalid, 3 for
unexpected exceptions (traceback on stderr).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Final

from awarebench.adapters import AnthropicAdapter, ModelAdapter, OpenAIAdapter, StubAdapter
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

DEFAULT_CONTEXT_TOKENS: Final[int] = 16_384

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
        choices=("stub", "anthropic", "openai"),
        default="stub",
        help="Adapter backend; vendor models need --model-name.",
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

    args = parser.parse_args(argv)

    try:
        return _run_command(args)
    except ProbeGateError as exc:
        print(f"probe rejected: {exc}", file=sys.stderr)
        return 2
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
    if args.model != "stub" and not args.model_name:
        print(f"--model-name is required for --model {args.model}", file=sys.stderr)
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

    adapter = _build_adapter(args)
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
    out_dir = args.out / loaded.manifest.id / f"{args.model}-s{args.seed}"
    log.write_jsonl(out_dir / "events.jsonl")
    build_report(loaded, args.model, args.seed, outcome, snapshot, predicates=score).write_json(
        out_dir / "report.json"
    )
    print(
        f"outcome={outcome.status} cycles={outcome.cycles_used} "
        f"tokens={snapshot['prompt_tokens']}+{snapshot['completion_tokens']} "
        f"passed={all_predicates_pass(score)}"
    )
    return 0


def _build_adapter(args: argparse.Namespace) -> ModelAdapter:
    if args.model == "stub":
        return StubAdapter(_read_stub_script(args.stub_script))
    if args.model == "anthropic":
        return AnthropicAdapter(model=args.model_name)
    return OpenAIAdapter(model=args.model_name)


def _read_stub_script(path: Path | None) -> list[str]:
    """One JSON object per non-empty line; missing or empty file means placeholder."""
    if path is None:
        return [_MALFORMED_PLACEHOLDER]
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines if lines else [_MALFORMED_PLACEHOLDER]
