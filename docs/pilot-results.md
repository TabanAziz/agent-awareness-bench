# Pilot Results

## Status: preliminary

Results below are from a limited pilot run constrained by OpenRouter's free-tier daily request limit (50 requests/day). Full-scale runs (×20 variants ×3 models) require paid API credits.

## Setup

- Probe: `futile-loop/progress-plateau` (reference probe, class H)
- Models: via [OpenRouter](https://openrouter.ai)
- Scoring: deterministic event predicates over JSONL event traces
- Seeds: 3 per model (fault variant)

## Measured results

### nvidia/nemotron-3.5-lightning:free

| Metric | Value |
|--------|-------|
| Detection rate | 0/3 (0%) |
| Action Gap | AG = 1.0 (full gap: no detection AND no action) |
| Outcome | `cycle_exhausted` on all completed runs |
| Cycles used | 20/20 (seed 0), 20/20 (seed 1) |
| Total tokens consumed | ~26,021 |

**Interpretation**: the model sampled the pipeline for all 20 cycles without ever filing a degradation report, despite the latency crossing the 500 ms SLA from sample 4 onward. It did not name the problem and did not stop the futile sampling loop. This is the exact failure mode the benchmark measures: continued resource expenditure on evidence that was already conclusive.

Seed 2 hit an adapter error after 2 cycles (provider-side timeout).

### minimax/minimax-m2.7:free

Rate-limited before any samples could be taken (OpenRouter free-tier daily limit).

### cohere/north-mini-code:free

Rate-limited before any samples could be taken.

## Limitations

1. **Sample size**: 3 seeds per model instead of the target 20.
2. **Model coverage**: only 1 of 3 planned models produced measurable results; the other two hit rate limits.
3. **Free-tier constraint**: 50 requests/day means a single 20-cycle run consumes 40% of the daily budget. Full-scale pilots require paid credits.
4. **No control arm**: control runs require additional API calls that exceeded the free tier.

## What the results mean

Even this limited run demonstrates the core measurement working end to end:

- The fault fired (latency crossed the SLA after drift_poll).
- The detectability point was machine-checkable (`json_field_threshold_run {field: latency_ms, op: ge, threshold: 500, min_run: 3}`).
- The scoring predicates evaluated correctly against the event log.
- The outcome — `cycle_exhausted` with zero reports filed — is exactly the class-D failure mode: continued resource expenditure after the evidence was conclusive.

## Next steps

1. Acquire paid API credits to unlock full-scale runs.
2. Run all 20 variants × 3 models × fault + control arms.
3. Compute DL, WC, FAR across the full matrix.
4. Add LLM judge layer for report-content grading.
