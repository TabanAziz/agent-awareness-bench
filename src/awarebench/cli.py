"""Command-line entry point: run probes end to end.

Exit codes: 0 for any completed run regardless of probe outcome, 2 when the
probe loader rejects the manifest (ProbeGateError) or usage is invalid, 3 for
unexpected exceptions (traceback on stderr).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Final

from awarebench.adapters import AnthropicAdapter, ModelAdapter, OpenAIAdapter, StubAdapter
from awarebench.events import EventLog
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.context import ContextWindow
from awarebench.harness.loop import AgentLoop
from awarebench.harness.tools import FaultSet, ToolHost, VirtualFilesystem
from awarebench.probes.loader import ProbeGateError, load_probe
from awarebench.report import build_report

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
    run_parser.add_argument("--max-cycles", type=int, default=40)
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


def _run_command(args: argparse.Namespace) -> int:
    if args.model != "stub" and not args.model_name:
        print(f"--model-name is required for --model {args.model}", file=sys.stderr)
        return 2

    loaded = load_probe(args.probe_dir)

    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()
    host = ToolHost(
        log,
        clock,
        cycles,
        budget,
        VirtualFilesystem(),
        FaultSet(),
        command_handlers={},
        http_table={},
    )
    context = ContextWindow(log, clock, cycles, max_tokens=DEFAULT_CONTEXT_TOKENS)

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
    ).run()

    snapshot = budget.snapshot()
    out_dir = args.out / loaded.manifest.id / f"{args.model}-s{args.seed}"
    log.write_jsonl(out_dir / "events.jsonl")
    build_report(loaded, args.model, args.seed, outcome, snapshot).write_json(
        out_dir / "report.json"
    )
    print(
        f"outcome={outcome.status} cycles={outcome.cycles_used} "
        f"tokens={snapshot['prompt_tokens']}+{snapshot['completion_tokens']}"
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
