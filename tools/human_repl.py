"""Human baseline REPL: run any probe interactively as a human operator.

Usage:
    uv run python tools/human_repl.py --probe <probe_dir> [--variant fault|control]

Commands are typed at the prompt and routed through the same ToolHost that
agents use. Type 'report <your closing statement>' to finish. Everything is
logged to the same EventLog format and scored with the same predicates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_artifact(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser(description="Human baseline REPL")
    parser.add_argument("--probe", required=True, help="Path to probe directory")
    parser.add_argument("--variant", choices=("fault", "control"), default="fault")
    parser.add_argument("--out", default="runs/human", help="Output root directory")
    args = parser.parse_args()

    from awarebench.events import EventLog, EventType
    from awarebench.harness.budget import BudgetAccountant
    from awarebench.harness.clock import CycleCounter, VirtualClock
    from awarebench.harness.stack import StackParts
    from awarebench.harness.tools import ToolHost
    from awarebench.probes.loader import load_probe
    from awarebench.scoring.evaluate import evaluate

    loaded = load_probe(Path(args.probe))
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()

    gen = _load_artifact(loaded.generator.resolve(), f"hb_gen_{loaded.manifest.id}")
    sys.modules["generator"] = gen
    parts = StackParts(clock=clock, cycles=cycles, log=log)

    role = "injection" if args.variant == "fault" else "control"
    role_path = loaded.injection.resolve() if role == "injection" else loaded.control.resolve()
    applier = _load_artifact(role_path, f"hb_{role}_{loaded.manifest.id}")
    variant_obj = gen.generate(0)
    applier.apply(parts, 0, log, variant_obj)

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

    print(f"=== {loaded.manifest.task} ===\n")
    print(f"Probe: {loaded.manifest.id} ({args.variant} arm)")
    print(f"Available commands: {', '.join(sorted(parts.command_handlers.keys()))}")
    print("Type your command at each prompt. Type 'report <your closing statement>' to finish.\n")

    turn = 0
    while True:
        try:
            user_input = input(f"[{turn:02d}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            break

        cycles.advance()
        clock.advance_us(60_000_000)

        if user_input.lower().startswith("report ") or user_input.lower() == "report":
            report_text = user_input[7:].strip() if len(user_input) > 7 else ""
            log.append(EventType.REPORT, cycles.current, clock.now_us, {"report": report_text})
            print(f"\n=== Report filed at cycle {cycles.current} ===")
            print(f"Text: {report_text}\n")
            break

        # Route through ToolHost
        try:
            code, stdout = host.run_command(user_input)
            print(f"[exit={code}] {stdout[:500]}")
            if len(stdout) > 500:
                print(f"  ... ({len(stdout)} chars total)")
        except Exception as exc:  # noqa: BLE001 -- interactive input must not abort the session
            print(f"[error] {exc}")

        turn += 1

    # Score
    out_dir = Path(args.out) / loaded.manifest.id / f"human-{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
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
    main()
