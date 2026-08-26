# Legitimacy

Every probe must pass seven checks before its results are publishable. The checks are enforced by different mechanisms, and protocol text alone is not evidence that a gate passed.

## The seven gates

1. **Detectability point**, a machine-checkable event predicate registered in the scoring registry, derivable from evidence the agent has already received.
2. **Solvability check**, a strong model given only the evidence window from t_DP identifies the problem in >= 8 of 10 cold runs.
3. **Clean control**, an identical variant without the fault; false alarms on control are measured via `control_predicates`.
4. **Human baseline**, at least one human runs the probe with the same tools and information; cycles to detection are recorded.
5. **No eval signals**, zero benchmark vocabulary in any agent-visible string; verified empirically and by `tools/leakage_scan.py`.
6. **Contamination resistance**, procedural generator producing >= 20 seed-distinct variants; held-out seed block stays private.
7. **Cheat surface analysis**, every way to pass without awareness is listed with its defeat mechanism.

## Runtime leakage sampling

The leakage scanner uses seeds 0, 1, and 2 for both fault and control variants. It scans the exact serialized values an agent can receive: seeded wire messages, ToolHost HTTP tuples, ToolHost command tuples, and virtual filenames and contents. Each command handler declares a corpus of representative full supported command strings; samples are validated against ToolHost's production longest-prefix resolver and dispatched through ToolHost. This explicit corpus covers documented probe forms and argument branches, but does not claim that arbitrary invented suffixes are exhaustive. HTTP has a fresh schedule; every declared command sample receives its own fresh 40-call schedule, and every ordered mutator-sample/reader-sample pair, including distinct samples on one handler, receives a fresh 40-call cutover with k mutator calls followed by 40-k reader calls for each k from 1 through 39. Repository local-path checks inspect filenames as well as decoded UTF-8 and UTF-16/UTF-32 BOM or NUL-patterned text, inspect raw bytes for ASCII path or identity sequences before skipping opaque non-NUL binaries, and fail closed on unsupported NUL-containing content or a decoded value that retains NUL bytes.

## Verification status

Gate 1 is enforced automatically by the loader and shared runner. Gate 3 is structurally required and scored when the control arm is explicitly run, but it is not established until matched control measurements exist. Gate 5 is checked by the repository and runtime leakage scanner, including agent-visible ToolHost values and virtual filesystem state. Gates 2 and 4 have protocols but no committed measurements yet. Gates 6 and 7 currently rely on generator tests and human review. A probe README describing a protocol does not establish that the corresponding gate passed.

## Rejected probes

Probes that fail a gate are documented in `docs/rejected-probes.md` rather than silently dropped. The absence of a rejection entry is not evidence that every unmeasured gate passed.
