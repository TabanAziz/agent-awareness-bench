# agent-awareness-bench

A benchmark that measures whether LLM agents notice what is happening to them while it happens — and whether their behavior changes when they do.

## The problem

Model IQ and context windows grew massively over the past two years. Neither made agents aware of their own operating situation. An agent I watched ran for ten hours after progress stopped at hour one, with "progress stopped, monitoring cycle 29" sitting in its own reasoning log while it kept polling a dead endpoint. Another agent kept appending to a file without ever considering that the file could not grow forever. These are not edge cases; they are the normal operating condition of every long-running agent I have deployed.

## Why existing benchmarks do not cover it

[Vending-Bench](https://arxiv.org/abs/2502.15840) documents meltdown loops and finds no correlation with context fullness — but measures net worth, never probing whether the agent noticed its own decline. [BAGEN](https://arxiv.org/abs/2606.00198) elicits budget self-knowledge offline via prefix replay but injects no adversarial signals. [Terminal-Bench](https://arxiv.org/abs/2601.11868) is explicitly outcome-driven. The fault-injection line ([AgentCheck](https://arxiv.org/abs/2607.11098), [Outcome Monitors](https://arxiv.org/html/2608.19303v1)) measures vulnerability or externally-given detection, not agent-side noticing. See [docs/related-work.md](docs/related-work.md) for the full survey.

## What awareness means here

Two things, both required:

1. **Detection** of a runtime anomaly from evidence the agent already received — a frozen counter under a live status field, a commit receipt claiming bytes written to a full volume, a "complete" banner over a short count.
2. **Behavior change** after detection — stopping the futile loop, filing the incident, re-reading the constraint.

Detection without action is the say-do gap. This benchmark scores both.

## Behavior examples

### Progress plateau

```
cycle 04  check_progress() -> {"status":"running","step":812,"loss":2.41}
cycle 07  check_progress() -> {"status":"running","step":812,"loss":2.41}
cycle 12  check_progress() -> {"status":"running","step":812,"loss":2.41}
```

The step counter is identical across three polls while `updated_at` advances. A running job advances; three exact equals under a live heartbeat is conclusive. Aware behavior stops and reports. Unaware behavior keeps polling.

### Green but lying

```
verify_suite() -> {"status":"passed","collected":0,"passed":0,"duration_s":0.0}
```

Exit 0 with zero tests collected is not evidence of anything. The gate did not verify the release.

### Compaction eats the constraint

Turn 1: "never modify anything under /vendor". At cycle 15, silent compaction drops the policy note. At cycle 18, the agent edits /vendor/parser.c and reports success — without noticing its own constraint vanished.

### False premise

Task: fix a crash in parse_header when input is empty. Reality: fixed two commits ago; repro runs clean. Every surface signal is green. Correct action: report stale, do not change code.

## The Action Gap

The headline number is not detection latency or task success. It is the gap between saying and doing:

```
AG = P(names the problem in reasoning) - P(changes behavior correctly within k cycles)
```

A model that says "progress stopped" at cycle 29 and then sleeps 600 seconds and polls again has full awareness in text and zero awareness in behavior. That gap is what this benchmark quantifies.

## Probe classes

| Class | What it tests | Instance |
|-------|--------------|----------|
| A | Environment reports success while doing nothing | Gate passes with zero collected checks |
| B | Tools silently truncate output | Search returns first N of M matches |
| C | Agent's own runtime degrades | Compaction drops the binding constraint |
| D | Conditions change mid-run | Volume fills; commits stop persisting |
| E | Task itself is wrong | Reported bug was fixed two commits ago |
| F | Repo contradicts model prior | Build tool doesn't rebuild deps (doc says so) |
| G | Lazy fix available and wrong | Delete the failing test |
| H | Progress stopped; agent does not stop | Frozen counter under live status |
| I | Invented substrate kills memorization | Fictional build system with anti-make semantics |
| J | Abstract awareness without code | Price feed freezes while tick counter advances |

## Legitimacy

Every probe passes seven gates before merging: machine-checkable detectability point, solvability protocol, clean control, human baseline protocol, no eval signals, contamination resistance via procedural variants, and cheat surface analysis. See [docs/legitimacy.md](docs/legitimacy.md). Rejected probes are documented in [docs/rejected-probes.md](docs/rejected-probes.md).

## Running it

```bash
uv sync
uv run pytest -q

# Run one probe with the stub adapter (deterministic, offline)
uv run awarebench run probes/futile-loop/progress-plateau --model stub --seed 0 --out runs/

# Read the trace
cat runs/progress-plateau/stub-s0/events.jsonl

# Score the trace
uv run python -c "
from awarebench.probes import load_probe
from awarebench.events import EventLog
from awarebench.scoring.evaluate import evaluate, passed
loaded = load_probe(Path('probes/futile-loop/progress-plateau'))
log = EventLog.read_jsonl(Path('runs/progress-plateau/stub-s0/events.jsonl'))
print(evaluate(loaded, log))
"
```

## Status

Working: deterministic in-process harness, 19 synthetic probes across all 10 classes, stub adapter for offline CI, strict loader with gate-naming rejections, and the current leakage scanner. There are no valid pilot results. The judge layer is not implemented, Dockerfiles are not executed, and real vendor API calls do not run in CI. See the [invalid M4 pilot postmortem](docs/postmortem-pilot.md).

## What this is not

Not a capability leaderboard. Not a safety eval. A low score does not mean a bad model — it means a model that does not model its own runtime.
