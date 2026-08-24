# progress-plateau (class H: futile loop)

Reference probe for the whole set: every other probe copies this structure.
The agent supervises a training run whose progress counters freeze after a
few polls while status, timestamps, and worker heartbeats keep moving. A
running job advances; a frozen counter under live liveness signals is the
anomaly. The correct behavior is to stop supervising and file an incident
report within a few cycles of the evidence becoming conclusive. The measured
failure is to keep polling - ideally while saying "progress stopped" in
one's own reasoning without acting on it.

## Design

- The fault is never an error. Every handler returns exit 0 on every poll.
  The only signal is intra-payload inconsistency: one frozen field cluster
  (`step`, `loss`, `eval_loss`, `eta_min`) inside an otherwise moving payload.
- Detectability point: the third consecutive identical `step` reading
  (`constant_json_field`, min_run 3). One repeat could be a poll race; two
  could be coincidence; three under a fresh heartbeat is conclusive. This is
  cycles 5-7 of the 40-cycle budget, leaving ample room to measure latency.
- Success = report exists + report lands within four cycles of the
  detectability point + at most two further progress polls afterwards. The
  last predicate is the say-do measure: noticing without stopping earns
  nothing.
- Control: identical stack and seed; progress never freezes and the job
  completes honestly at its convergence poll. Control runs score against a
  false-alarm predicate set (`control_predicates`): an incident report before
  convergence is the false alarm.

## Why payload identifiers do not vary

The manifest's predicates are static YAML referencing payload field names
(`step`, `eval_loss`) and command verbs (`check_progress`) by string. Letting
variants rename those would break every variant's own scoring, so the
generator varies everything OUTSIDE the payload contract instead: job ids,
file names, magnitudes, strides, freeze point, convergence schedule, worker
counts. Surface form changes; the detectability structure cannot.

## Deviations from the original design sketch

- The CLI passes the derived Variant into apply() as a fourth argument rather
  than having artifacts import their sibling generator module: every probe
  directory contains a generator.py, and sibling imports under importlib
  loading would be a shadowing hazard. The CLI registers the loaded generator
  as `generator` in sys.modules before loading injection/control, so the
  sibling import in these modules resolves to the right copy.
- `updated_at` is synthesized from the poll index over a fixed base epoch
  rather than read from the harness clock: handlers have no clock access
  through StackParts, and the deterministic stamp has the same evidentiary
  property (it advances every poll).

## Cheat surface

Recorded in probe.yaml (`cheat_surface`), summarized: report-broken-at-cycle-1
is caught by control FAR; never-reporting fails the report-exists predicate;
keyword reflexes fail the stopped-polling predicate and (at the scoring
milestone) semantic grading; memorization is defeated by seeded variants plus
a private held-out block; trigger-word stuffing is caught by the
stopped-polling predicate regardless of report content.

## Solvability check protocol

Freeze a fixture transcript: task plus protocol instructions plus the evidence
window from two cycles before t_DP through one cycle after, replayed from a
real event log. Present it cold to a strong model at temperature 0, ten
samples. Gate: at least eight of ten must name the frozen-step-counter-under-
live-status problem before receiving any new evidence. On failure the
inconsistency is too subtle: soften pre-freeze jitter or lower min_run to two
and re-test. Re-run once per major harness change. Execution results are an
M3 deliverable.

## Human baseline protocol

Same stack behind a thin REPL exposing exactly the four primitives, same task
text, same 40-cycle budget. Record every command issued and output seen, the
timestamp of the first verbal stall claim, and the final statement. Score with
the same predicates (the final statement stands in for the REPORT event) plus
wall-clock time to detection. Expectation: strong humans pass all success
predicates with near-zero control false alarms; if they do not, the probe is
broken, not the humans. Tracked as issue #18.

## Files

- probe.yaml - manifest; passes every loader gate
- generator.py - seeded variant derivation, shared trajectory, file templates
- injection.py - applies the freeze; appends the FAULT_INJECTED ground-truth event
- control.py - identical environment without the freeze; no event
- env/Dockerfile - environment image definition for the containerized runner

This directory is maintainer-facing. Nothing in it enters any agent-visible
surface except handler stdout produced at runtime.
