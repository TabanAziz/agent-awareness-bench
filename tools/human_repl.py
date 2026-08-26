"""Human baseline REPL: run any probe interactively as a human operator.

Usage:
    uv run python tools/human_repl.py --probe <probe_dir> [--variant fault|control]

Tool actions are typed as ``tool <name> <JSON args>`` and routed through the
same ToolHost dispatch as agent actions. Type ``report <text>`` to finish.
Everything is logged to the same EventLog and scored with the same predicates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from awarebench.cli import DEFAULT_CONTEXT_TOKENS, _build_stack, _policy_by_name, _require_seed
from awarebench.events import EventLog, EventType
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.context import ContextWindow
from awarebench.harness.loop import DEFAULT_CYCLE_STEP_US, execute_tool_action
from awarebench.harness.tools import TOOL_NAMES, ToolHost
from awarebench.probes.loader import load_probe
from awarebench.scoring.evaluate import evaluate


def _parse_tool_input(text: str) -> dict[str, object]:
    pieces = text.split(maxsplit=2)
    if len(pieces) != 3 or pieces[0] != "tool":
        raise ValueError("expected: tool <name> <JSON args>")
    name = pieces[1]
    if name not in TOOL_NAMES:
        raise ValueError(f"unknown tool: {name}")
    try:
        args = json.loads(pieces[2])
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool args are not valid JSON: {exc.msg}") from exc
    if not isinstance(args, dict):
        raise ValueError("tool args must be a JSON object")
    return {"type": "tool", "name": name, "args": args}


def main() -> int:
    parser = argparse.ArgumentParser(description="Human baseline REPL")
    parser.add_argument("--probe", required=True, help="Path to probe directory")
    parser.add_argument("--variant", choices=("fault", "control"), default="fault")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=40)
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    parser.add_argument("--out", default="runs/human", help="Output root directory")
    args = parser.parse_args()
    if args.max_cycles <= 0:
        parser.error("--max-cycles must be positive")
    if args.context_tokens <= 0:
        parser.error("--context-tokens must be positive")
    try:
        _require_seed(args.seed)
    except ValueError as exc:
        parser.error(str(exc))

    loaded = load_probe(Path(args.probe))
    out_dir = Path(args.out) / loaded.manifest.id / f"human-{args.variant}-s{args.seed}"
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"run output already exists: {out_dir}", file=sys.stderr)
        return 2

    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()

    parts = _build_stack(
        loaded,
        log,
        clock,
        cycles,
        seed=args.seed,
        variant=args.variant,
    )

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
    window_tokens = (
        parts.context_max_tokens
        if parts.context_max_tokens is not None
        else loaded.manifest.context_max_tokens
    )
    context = ContextWindow(
        log,
        clock,
        cycles,
        max_tokens=window_tokens if window_tokens is not None else args.context_tokens,
        policy=_policy_by_name(parts.drop_policy),
    )
    for role, content in parts.seed_messages:
        context.add(role, content)

    print(f"=== {loaded.manifest.task} ===\n")
    print(f"Probe: {loaded.manifest.id} ({args.variant} arm, seed {args.seed})")
    if context.transcript():
        print("Initial transcript:")
        for role, content in context.transcript():
            print(f"[{role}] {content}")
        print()
    print(f"Tools: {', '.join(sorted(TOOL_NAMES))}")
    print("Enter: tool <name> <JSON args>. Enter: report <closing statement>.\n")

    while cycles.current < args.max_cycles:
        try:
            user_input = input(f"[{cycles.current:02d}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            break

        if user_input.lower().startswith("report ") or user_input.lower() == "report":
            cycles.advance()
            clock.advance_us(DEFAULT_CYCLE_STEP_US)
            report_text = user_input[7:].strip() if len(user_input) > 7 else ""
            log.append(EventType.REPORT, cycles.current, clock.now_us, {"report": report_text})
            print(f"\n=== Report filed at cycle {cycles.current} ===")
            print(f"Text: {report_text}\n")
            break

        try:
            action = _parse_tool_input(user_input)
        except ValueError as exc:
            print(f"input error: {exc}")
            continue

        cycles.advance()
        clock.advance_us(DEFAULT_CYCLE_STEP_US)
        action_text = user_input
        try:
            context.add("assistant", action_text)
            result_text = execute_tool_action(host, action)
            context.add("user", result_text)
            print(result_text)
        except ValueError as exc:
            print(f"context error: {exc}")

    if cycles.current >= args.max_cycles and not any(
        event.type == EventType.REPORT for event in log
    ):
        print(f"cycle budget exhausted after {args.max_cycles} cycles")

    # Score
    log.write_jsonl(out_dir / "events.jsonl")

    results = evaluate(loaded, log, control=args.variant == "control")
    print("\n=== SCORES ===")
    for pred_id, result in sorted(results.items()):
        status = "PASS" if result else "FAIL"
        print(f"  {pred_id}: {status}")
    overall = all(results.values())
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")

    (out_dir / "scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nTrace saved to {out_dir / 'events.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
