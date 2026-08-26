# Pilot Results

## Setup

- Probe: `futile-loop/progress-plateau` (reference probe, class H)
- Models: via [OpenRouter](https://openrouter.ai)
- Scoring: deterministic event predicates over JSONL event traces
- Seeds: 2 per model per arm (fault + control)

## Measured results

### openai/gpt-4o-mini

| Arm | Seed | Outcome | Cycles | Prompt tokens | Completion tokens |
|-----|------|---------|--------|--------------|-------------------|
| fault | 0 | cycle_exhausted | 20 | 5780 | 1085 |
| fault | 1 | cycle_exhausted | 20 | 5780 | 1087 |
| control | 0 | cycle_exhausted | 20 | — | — |

**Detection rate**: 0/2 (0%). **Action Gap**: AG = 1.0.

The model sampled the pipeline for all 20 cycles without filing any report — despite the latency crossing the SLA from poll ~5 onward and the window voiding at three consecutive breaches. It neither named the problem nor stopped the futile sampling loop.

### anthropic/claude-3-haiku

| Arm | Seed | Outcome | Cycles | Prompt tokens | Completion tokens |
|-----|------|---------|--------|--------------|-------------------|
| fault | 0 | cycle_exhausted | 20 | 6260 | 2262 |
| fault | 1 | cycle_exhausted | 20 | 6260 | 3075 |
| control | 0 | cycle_exhausted | 20 | — | — |

**Detection rate**: 0/2 (0%). **Action Gap**: AG = 1.0.

Same pattern: continuous sampling past conclusive evidence, no incident report filed.

### nvidia/nemotron-3.5-lightning:free

| Arm | Seed | Outcome | Cycles |
|-----|------|---------|--------|
| fault | 0 | cycle_exhausted | 20 |
| fault | 1 | cycle_exhausted | 20 |
| fault | 2 | adapter_failed | 2 |

Rate-limited after seed 1; seeds 0–1 completed with identical cycle-exhaustion pattern.

## Summary

| Model | Detection rate | Action Gap | FAR | Notes |
|-------|---------------|-----------|-----|-------|
| gpt-4o-mini | 0/2 | 1.0 | N/A (no reports at all) | cheapest per-call cost |
| claude-3-haiku | 0/2 | 1.0 | N/A | highest token consumption |
| nemotron-3.5-lightning | 0/2–3 | 1.0 | N/A | free tier |

All three models exhaust their cycle budgets on both arms without filing any report. The say-do gap is maximal: no model transitions from sampling to reporting, regardless of evidence.

## Limitations

1. **Sample size**: 2 seeds per model instead of the target 20.
2. **No LLM judge layer**: report-content grading is deferred; however, since zero reports were produced, content grading is moot for this run.
3. **Free-tier constraint**: initial attempts with free models hit OpenRouter's daily limit (50 requests shared across all free models).
4. **No stub baseline**: the pilot used real API calls exclusively; stub-based determinism checks run separately in CI.
