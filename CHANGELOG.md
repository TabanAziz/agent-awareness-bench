# Changelog

## 0.1.0

### Added

- Full deterministic harness: event log, virtual clock, cycle counter, budget accountant, context window with silent compaction, lying tool primitives, model adapters (stub/anthropic/openai/openrouter).
- Single benchmark agent loop through `AgentLoop`, with a shared adapter-owned completion gateway, fixed-payload regression guards, collision-resistant arm/model run identities, preserved reasoning provenance, and no model I/O outside adapters.
- Strict probe manifest loader: rejects any probe missing a detectability point, control variant, procedural generator, or cheat-surface analysis.
- 19 probes across all 10 classes (A through J), each with generator, injection, control, and end-to-end tests.
- 22+ machine-checkable scoring predicates over event logs.
- CLI: `awarebench run <probe> --model <model> --seed <seed>`.
- Two-model semantic judge with per-probe rubrics, duplicate-key-safe strict
  JSON decisions, distinct stable provider-resolved identities, preserved
  disagreements and transport metadata, exact `t_DP` ordering,
  detection-relative deterministic action predicates, full-log prerequisite
  context, explicit unavailable action semantics, and `awarebench judge`
  durable output.
- Detection-rate, detection-latency, Action Gap, false-alarm, and judge
  disagreement aggregation with explicit unavailable semantics.
- Judge validation capture/replay infrastructure for two isolated adapter calls
  per excerpt. Release remains gated on human sign-off and fresh captured
  outputs for at least 50 held-out excerpts at 90% agreement.
- Leakage scanner (`tools/leakage_scan.py`).
- CI workflow: lint, typecheck, test, leakage scan, determinism check.
