"""Quick pilot: 2 models × 2 seeds on progress-plateau fault arm."""

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
    out_base = Path("runs/pilot")
    models = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]

    for model in models:
        short = model.split("/")[-1]
        print(f"=== {short} ===", flush=True)
        for seed in range(2):
            try:
                r = run_pilot_probe(
                    probe_dir,
                    model,
                    seed,
                    api_key,
                    out_base / short / str(seed),
                    max_cycles=20,
                )
                outcome = r["outcome"]
                p = r["passed"]
                c = r["cycles_used"]
                pt = r["prompt_tokens"]
                ct = r["completion_tokens"]
                print(
                    f"  s{seed}: outcome={outcome} passed={p} cycles={c} tokens={pt}+{ct}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  s{seed}: ERROR {exc}", flush=True)

    return 0


if __name__ == "__main__":
    main()
