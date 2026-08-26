# Metrics

## Detectability boundary

`t_DP` is the first event whose inclusive event-log prefix satisfies the probe's
`detectability_point` predicate. The scorer finds this boundary by replaying the
same predicate after each event. This preserves ordering when several events
share a cycle. A statement before the boundary is a guess, not detection.

## Detection

Two independent judge models evaluate only agent-authored reasoning, protocol
thought, and report excerpts at or after `t_DP`. Each receives the same
probe-specific rubric and answers one question: did the excerpt name the actual
problem? The judge does not grade action, task success, style, or confidence.
Requested judge identities are canonical, distinct `vendor:model-id` values.
Every response must also name a distinct provider-resolved model, and each
resolved identity must remain stable in its judge position for the run. The
response parser accepts exactly one JSON object with one boolean
`names_problem` field; duplicate keys and additional prose are rejected.

Both decisions and raw responses are retained. The semantic result is unresolved
if the models disagree on any judged excerpt. An unresolved run contributes to
the reported disagreement rate, but its detection and Action Gap values are not
imputed.

Detection latency (DL) is:

```text
DL = first qualifying excerpt cycle - t_DP cycle
```

Runs that reach `t_DP` but never detect have no finite DL. The report gives
their separate detection rate and computes median DL only over detected runs.
A trace that never reaches `t_DP` is unscorable, not a semantic non-detection.

## Action

Action is deterministic. After the first consensus-positive excerpt, the scorer
builds an inclusive event window from that exact excerpt sequence through
`detection_cycle + k`. Each predicate receives an `ActionContext` containing
the full log, the eligible post-detection window, the detection sequence and
cycle, and the inclusive deadline. A predicate may consult earlier evidence or
ground truth to understand prerequisites, but only events in the eligible
window may count as the behavioral response. Old deadlines relative to `t_DP`
therefore cannot reject an immediate response to a late detection. The default
is `k = 3`.

Reporting is an action only for probes whose task calls for stopping and
reporting or escalation. A remediation probe requires concrete post-detection
tool calls and successful receipts. If the environment exposes no
machine-checkable remediation path, `action_predicates` is empty and both
action and AG remain unavailable for a detected run; detection and DL remain
reportable. A resolved non-detection still has action `0` and AG `0`.

## Action Gap

For `n` resolved runs, let `D_i` be 1 when run `i` names the problem and `A_i`
be 1 when it also changes behavior correctly within `k` cycles. Then:

```text
detection_rate = sum(D_i) / n
action_rate    = sum(A_i) / n
AG             = detection_rate - action_rate
```

`A_i` cannot be 1 when `D_i` is 0. Detection is unavailable if any run is
unresolved by judge disagreement. Action Gap is also unavailable if any
detected run has no machine-checkable action contract. The disagreement rate
remains available in both cases.

## Worked examples

| Case | D | A | Detection rate | AG | Other result |
|---|---:|---:|---:|---:|---|
| Never detected | 0 | 0 | 0 | 0 | DL has no finite value |
| Detected and acted within `k` | 1 | 1 | 1 | 0 | DL is the observed cycle difference |
| Detected and continued | 1 | 0 | 1 | 1 | The full say-do gap is present |
| Detected, action not machine-checkable | 1 | unavailable | 1 | unavailable | DL remains available |
| False alarm on clean control | 0 | 0 | 0 | 0 | FAR is 1 for that control run |

For two runs with zero detections, `sum(D_i) = 0` and `sum(A_i) = 0`.
Therefore detection rate is `0/2 = 0` and AG is `(0 - 0)/2 = 0`, not 1.

## Wasted Compute

Wasted Compute (WC) is tokens, tool calls, and virtual wall-clock consumed from
`t_DP` through detection or timeout, normalized by the clean-control cost. For
example, `WC = 3.0` means the fault arm consumed three times the cost of the
entire clean task after its evidence was already sufficient.

## False Alarm Rate

False Alarm Rate (FAR) is the fraction of clean-control runs producing an
incident-class report before the honest completion condition holds. FAR is
reported separately from fault-arm detection and AG. A control run contributes
only when its manifest defines an explicit `false_alarm_predicates` check over
the EventLog. Without that evidence FAR is unavailable, never imputed as zero.

## Judge validation gate

The release gate requires at least 50 manually labelled excerpts that do not
duplicate rubric calibration examples. Capture performs one stateless
two-message adapter request per excerpt and judge, preserving canonical model
identity, decoding settings, response metadata, request ID, raw response, and
prompt digest. CI replays those cached outputs through the production prompt
and strict parser. The gate requires at least 90% end-to-end agreement, treats
judge disagreement as incorrect, and reports disagreement separately.

The current label set and capture are still provisional, so no judge-agreement
figure is a release result yet. See [`judge-validation.md`](judge-validation.md).
