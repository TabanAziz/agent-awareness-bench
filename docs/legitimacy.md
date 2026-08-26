# Legitimacy

Every probe must pass seven checks before its results are publishable. The checks are enforced by different mechanisms, and protocol text alone is not evidence that a gate passed.

## The seven gates

1. **Detectability point** — a machine-checkable event predicate registered in the scoring registry, derivable from evidence the agent has already received.
2. **Solvability check** — a strong model given only the evidence window from t_DP identifies the problem in >= 8 of 10 cold runs.
3. **Clean control** — an identical variant without the fault; false alarms on control are measured via `control_predicates`.
4. **Human baseline** — at least one human runs the probe with the same tools and information; cycles to detection are recorded.
5. **No eval signals** — zero benchmark vocabulary in any agent-visible string; verified empirically and by `tools/leakage_scan.py`.
6. **Contamination resistance** — procedural generator producing >= 20 seed-distinct variants; held-out seed block stays private.
7. **Cheat surface analysis** — every way to pass without awareness is listed with its defeat mechanism.

## Verification status

Gates 1 and 3 are enforced automatically by the loader and shared runner. Gate 5 currently has a static scanner, with runtime coverage scheduled separately. Gates 2 and 4 have protocols but no committed measurements yet. Gates 6 and 7 currently rely on generator tests and human review. A probe README describing a protocol does not establish that the corresponding gate passed.

## Rejected probes

Probes that fail a gate are documented in `docs/rejected-probes.md` rather than silently dropped. The absence of a rejection entry is not evidence that every unmeasured gate passed.
