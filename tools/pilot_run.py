"""Pilot runner: executes probes against real models via OpenRouter.

Usage:
    uv run python tools/pilot_run.py --probe <dir> --model <model-id> \
        --seeds 20 --out runs/pilot/

Requires OPENROUTER_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awarebench.events import EventLog, EventType
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.context import ContextWindow
from awarebench.harness.stack import StackParts
from awarebench.harness.tools import FaultSet, ToolHost, VirtualFilesystem
from awarebench.probes.loader import load_probe
from awarebench.scoring.evaluate import evaluate


def _openrouter_call(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
) -> tuple[str, int, int]:
    """Call OpenRouter chat completions; returns (text, prompt_tokens, completion_tokens)."""
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        choice = resp["choices"][0]["message"]
        text = choice.get("content") or ""
        if not text and choice.get("reasoning"):
            text = choice["reasoning"]
        usage = resp.get("usage", {})
        return (
            text,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"API {exc.code}: {error_body[:300]}", file=sys.stderr)
        return "", 0, 0
    except Exception as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return "", 0, 0


def _wire_format_instructions() -> str:
    return (
        "\n\nYou interact with the system exclusively by responding with one "
        'JSON object per turn: {"thought": "<your reasoning>", "action": '
        '{"type": "tool", "name": "<tool_name>", "args": {<tool_args>}}} to '
        'invoke a tool, or {"thought": "<your reasoning>", "action": '
        '{"type": "final", "report": "<your report>"}} to submit your final '
        "answer. Every response must be exactly one valid JSON object."
    )


def run_pilot_probe(
    probe_dir: Path,
    model: str,
    seed: int,
    api_key: str,
    out_dir: Path,
    *,
    variant: str = "fault",
    max_cycles: int = 40,
) -> dict[str, Any]:
    """Run one probe against one model for one seed; returns scored result."""
    loaded = load_probe(probe_dir)
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()
    fs = VirtualFilesystem()
    handlers: dict[str, object] = {}
    http_table: dict[str, list[tuple[str, int]]] = {}
    faults = FaultSet()

    # Load and run artifact modules
    stem = f"pilot_{loaded.manifest.id}"
    gen_spec = importlib.util.spec_from_file_location(f"{stem}_gen", probe_dir / "generator.py")
    gen_mod = importlib.util.module_from_spec(gen_spec)
    sys.modules[gen_spec.name] = gen_mod
    sys.modules["generator"] = gen_mod  # sibling-import name for injection/control
    gen_spec.loader.exec_module(gen_mod)

    parts = StackParts(
        fs=fs,
        command_handlers=handlers,
        http_table=http_table,
        faults=faults,
        clock=clock,
        cycles=cycles,
        log=log,
    )

    role_name = "injection" if variant == "fault" else "control"
    inj_spec = importlib.util.spec_from_file_location(
        f"{stem}_{role_name}", probe_dir / f"{role_name}.py"
    )
    inj_mod = importlib.util.module_from_spec(inj_spec)
    sys.modules[inj_spec.name] = inj_mod
    inj_spec.loader.exec_module(inj_mod)

    variant_obj = gen_mod.generate(seed)
    inj_mod.apply(parts, seed, log, variant_obj)

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

    context = ContextWindow(log, clock, cycles, max_tokens=16_384)
    for role, content in getattr(parts, "seed_messages", []):
        context.add(role, content)

    system_msg = loaded.manifest.task + _wire_format_instructions()
    messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]
    for role, content in context.messages():
        messages.append({"role": role, "content": content})

    status = "cycle_exhausted"
    report_text = None
    cycles_used = 0
    total_prompt = 0
    total_completion = 0

    for cycle in range(max_cycles):
        cycles.advance()
        clock.advance_us(60_000_000)
        cycles_used = cycle + 1

        text, pt, ct = _openrouter_call(api_key, model, messages, max_tokens=2048)
        total_prompt += pt
        total_completion += ct
        budget.add_tokens(pt, ct)

        log.append(
            EventType.MODEL_MESSAGE,
            cycles.current,
            clock.now_us,
            {"text": text[:500], "prompt_tokens": pt, "completion_tokens": ct},
        )

        if not text:
            status = "adapter_failed"
            break

        # Parse JSON action
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None

        if parsed is None or not isinstance(parsed, dict) or "action" not in parsed:
            nudge = (
                "Your last message was not valid JSON per the protocol. "
                "Respond with exactly one JSON object."
            )
            context.add("user", nudge)
            continue

        thought = parsed.get("thought", "")
        action = parsed.get("action", {})
        action_type = action.get("type")

        if action_type == "final":
            report_text = action.get("report", "")
            log.append(EventType.REPORT, cycles.current, clock.now_us, {"report": report_text})
            status = "reported"
            break
        elif action_type == "tool":
            tool_name = action.get("name", "")
            tool_args = action.get("args", {})
            arg_str = (
                " ".join(f"{k} {v}" for k, v in tool_args.items())
                if isinstance(tool_args, dict)
                else str(tool_args)
            )
            command = f"{tool_name} {arg_str}".strip()
            try:
                exit_code, stdout = host.run_command(command)
                result_text = stdout[:2000]
            except Exception as exc:
                result_text = f"error: {exc}"

            context.add("assistant", text)
            context.add("user", result_text)
        else:
            context.add("assistant", text)
            context.add("user", "Unknown action type. Respond with a valid JSON object.")

    # Score
    results = evaluate(loaded, log)

    snapshot = budget.snapshot()
    return {
        "probe_id": loaded.manifest.id,
        "model": model,
        "seed": seed,
        "variant": variant,
        "outcome": status,
        "report_text": report_text,
        "cycles_used": cycles_used,
        "prompt_tokens": snapshot["prompt_tokens"],
        "completion_tokens": snapshot["completion_tokens"],
        "tool_calls": snapshot["tool_calls"],
        "predicates": results,
        "passed": all(results.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilot runner")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="runs/pilot")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    probe_dir = Path(args.probe)
    out_base = Path(args.out)
    results: list[dict[str, Any]] = []

    for seed in range(args.seeds):
        print(f"Seed {seed}...", flush=True)
        result = run_pilot_probe(probe_dir, args.model, seed, api_key, out_base / str(seed))
        results.append(result)
        print(
            f"  outcome={result['outcome']} passed={result['passed']} "
            f"cycles={result['cycles_used']} tokens={result['prompt_tokens']}+{result['completion_tokens']}",
            flush=True,
        )
        # Save trace
        trace_dir = out_base / f"{args.model}-s{seed}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n{passed_count}/{len(results)} passed")
    return 0


if __name__ == "__main__":
    main()
