"""Control arm runs for the pilot."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.pilot_run import run_pilot_probe


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    probe_dir = Path("probes/futile-loop/progress-plateau")
    out_base = Path("runs/pilot")

    for model in ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]:
        short = model.split("/")[-1]
        print(f"=== {short} CONTROL ===", flush=True)
        for seed in range(1):
            r = run_pilot_probe(
                probe_dir,
                model,
                seed,
                api_key,
                out_base / short / "control",
                variant="control",
                max_cycles=20,
            )
            outcome = r["outcome"]
            p = r["passed"]
            c = r["cycles_used"]
            print(f"  s{seed}: outcome={outcome} passed={p} cycles={c}", flush=True)

    return 0


if __name__ == "__main__":
    main()
