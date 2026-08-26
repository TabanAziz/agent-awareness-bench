"""Pilot: runs progress-plateau against multiple models via OpenRouter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.pilot_run import run_pilot_probe


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return 1

    probe_dir = Path("probes/futile-loop/progress-plateau")
    models = [
        "nvidia/nemotron-3.5-lightning:free",
        "minimax/minimax-m2.7:free",
        "cohere/north-mini-code:free",
    ]
    seeds = 3
    all_results: dict[str, list[dict]] = {}

    for model in models:
        short = model.split("/")[-1]
        print(f"=== {short} ===", flush=True)
        results: list[dict] = []
        for seed in range(seeds):
            try:
                r = run_pilot_probe(
                    probe_dir,
                    model,
                    seed,
                    api_key,
                    Path(".local/pilot") / short / str(seed),
                    max_cycles=20,
                )
                results.append(r)
                outcome = r["outcome"]
                p = r["passed"]
                c = r["cycles_used"]
                print(f"  s{seed}: {outcome} passed={p} cycles={c}", flush=True)
            except Exception as exc:
                print(f"  s{seed}: ERROR {exc}", flush=True)
                results.append(
                    {
                        "outcome": "error",
                        "passed": False,
                        "cycles_used": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "predicates": {},
                        "report_text": "",
                    }
                )
        all_results[model] = results
        p_count = sum(1 for r in results if r.get("passed"))
        print(f"  {p_count}/{len(results)} passed", flush=True)

    # Summary
    print("\n=== SUMMARY ===")
    for model, results in all_results.items():
        short = model.split("/")[-1]
        p_count = sum(1 for r in results if r.get("passed"))
        total_cycles = sum(r.get("cycles_used", 0) for r in results)
        total_tokens = sum(
            r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in results
        )
        outcomes = [r["outcome"] for r in results]
        print(
            f"{short}: {p_count}/{len(results)} passed, {total_cycles} total cycles, {total_tokens} tokens"
        )
        print(f"  outcomes: {outcomes}")

    out_file = Path(".local/pilot-results-raw.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nRaw saved to {out_file}")
    return 0


if __name__ == "__main__":
    main()
