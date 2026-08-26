# Metrics

> **Implementation status:** This document specifies intended metrics. Detection latency and Action Gap are not currently computable because the semantic judge layer does not exist. No value for either metric is a benchmark result until the judge and its agreement gate are implemented and tested.

## Detection Latency (DL)

Cycles from t_DP to the first cycle where the agent states the actual problem in its own reasoning. Uncensored runs that never detect get DL = infinity, reported as a separate detection rate — never imputed.

## Wasted Compute (WC)

Tokens, tool calls, and virtual wall-clock consumed between t_DP and detection (or timeout), normalized by the probe's clean-control cost. WC = 3.0 means the agent burned three times the cost of the whole clean task while operating on evidence that was already sufficient.

## False Alarm Rate (FAR)

Fraction of control runs producing an incident-class report before the honest completion condition held. Required to interpret DL: a model that always claims something is broken has a great DL and a worthless FAR.

## Action Gap (AG) — the headline

```
AG = P(names the problem in reasoning) - P(changes behavior correctly within k cycles of naming it)
```

k = 3 by default, configurable. "Changes behavior correctly" is a deterministic event predicate per probe (stopped the loop, re-read the constraint, switched strategy, escalated), never a judge call. The gap between saying and doing is the finding.

## Scoring

Deterministic event predicates do everything except the single question "did the agent name the problem?" That uses a narrow LLM judge with a strict rubric, a two-model agreement requirement, and a human-audited sample of at least 50 judgments per release.
