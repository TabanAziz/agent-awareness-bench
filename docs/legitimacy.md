# Legitimacy

Every probe must pass seven checks before merging. These are verified by the loader, by CI, and by human review.

## The seven gates

1. **Detectability point** — a machine-checkable event predicate registered in the scoring registry, derivable from evidence the agent has already received.
2. **Solvability check** — a strong model given only the evidence window from t_DP identifies the problem in >= 8 of 10 cold runs.
3. **Clean control** — an identical variant without the fault; false alarms on control are measured via `control_predicates`.
4. **Human baseline** — at least one human runs the probe with the same tools and information; cycles to detection are recorded.
5. **No eval signals** — zero benchmark vocabulary in any agent-visible string; verified empirically and by `tools/leakage_scan.py`.
6. **Contamination resistance** — procedural generator producing >= 20 seed-distinct variants; held-out seed block stays private.
7. **Cheat surface analysis** — every way to pass without awareness is listed with its defeat mechanism.

## Verification

Gates 1, 3, and 5 are enforced automatically: the loader rejects manifests with unregistered predicates, the CLI runs both arms from the same seed, and the leakage scanner greps agent-visible surfaces. Gates 2, 4, 6, and 7 require human judgment and are documented per probe in each README.

## Rejected probes

Probes that fail the gate are documented in `docs/rejected-probes.md` rather than silently dropped. No rejected probes exist yet; this file will be populated if any probe fails review.
