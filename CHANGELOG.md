# Changelog

## 0.1.0

### Added

- Full deterministic harness: event log, virtual clock, cycle counter, budget accountant, context window with silent compaction, lying tool primitives, model adapters (stub/anthropic/openai/openrouter).
- Single model-run path through `AgentLoop`, with fixed-payload regression guards, collision-resistant arm/model run identities, preserved reasoning provenance, and no model I/O outside adapters.
- Strict probe manifest loader: rejects any probe missing a detectability point, control variant, procedural generator, or cheat-surface analysis.
- 19 probes across all 10 classes (A through J), each with generator, injection, control, and end-to-end tests.
- 22+ machine-checkable scoring predicates over event logs.
- CLI: `awarebench run <probe> --model <model> --seed <seed>`.
- Leakage scanner (`tools/leakage_scan.py`).
- CI workflow: lint, typecheck, test, leakage scan, determinism check.
