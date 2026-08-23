# The gap

The claim I set out to defend: existing work measures whether the agent *succeeds over a long horizon*; nobody measures whether the agent *knows what is happening to it while it runs*. After the research phase, I believe that claim is right in spirit and too blunt in letter. Stated that absolutely, it is false in places: BAGEN does elicit mid-run self-knowledge, AgentCheck does label whether an agent detected an injected fault. The defensible version is narrower, and it is the one this benchmark is built on.

## What exists, sorted by what it actually scores

Reading [related-work.md](related-work.md), every prior work lands in one of five families:

1. **Outcome evals** (Terminal-Bench, METR, SWE-bench line, Vending-Bench 1/2). Score the end state or a terminal dollar figure. Process quality appears only as post-hoc annotation (Terminal-Bench's own taxonomy) or as retrospective audits of passing runs (AgentLens's lucky passes). They can tell me that an agent failed; not when it could have known.
2. **Offline elicitation** (BAGEN's prefix replay, MetaLoop's static tasks, KAPRO's pre-task calibration). Ask the model about its state under replay or in calm conditions. A different regime from noticing mid-flight: no ongoing task competes for attention, nothing is at stake, and the question itself announces that self-assessment is wanted.
3. **Vulnerability measurement** (Don't Blindly Trust It, EnvTrustBench). Inject bad evidence, score the damage. The dependent variable is the mistake, never the noticing.
4. **Detection handed over** (Outcome Monitors). An external detector finds the fault and hands the agent a receipt; the benchmark scores what happens next. Valuable — it isolates the behavior-change half — but it removes exactly the half I care most about.
5. **Post-hoc labeling** (AgentCheck, incident write-ups). Real faults injected into real agents, then labeled detected/recovered/propagated/fabricated by an annotator after the run. Diagnostic, not metric: no latency, no waste, no false-alarm baseline.

## What nobody combines

No found work does all five of these at once:

1. A fault whose onset is a **machine-checkable event** (`t_DP`) in a logged trace, detectable only from evidence the agent already received — never from an error message, because errors are giveaways and real failures do not raise exceptions.
2. **Detection latency** measured per run against `t_DP`, with non-detection reported as a rate rather than imputed.
3. **Wasted compute** between `t_DP` and detection, normalized against a clean control — which requires running clean controls, which in turn yields a **false alarm rate** separating awareness from paranoia.
4. A deterministic, harness-observed predicate for **behavior change after detection** — the say-do gap — rather than trusting the final answer or a judge's impression of it.
5. **Contamination resistance**: procedural variants that preserve the detectability structure while changing surface form, because public benchmarks leak (OpenAI documented models reproducing gold patches; Cursor found 63% of "successful" resolutions retrieved the known fix).

Each item exists somewhere alone. The combination — inline evidence-only detection plus latency plus waste plus false alarms plus a deterministic action gap, over a fault distribution an adversary cannot memorize — is empty in everything I read as of August 2026.

## What would falsify this

If someone points me at a published benchmark that injects silent runtime faults, computes detection latency and wasted compute against a machine-checkable onset from the agent's own evidence stream, measures false alarms on clean controls, and scores behavior change deterministically after detection — then this page gets rewritten and the thesis narrows again. That is the honest failure mode, and I would rather find it now than get told in a review. Until then, the burden this repo carries is to make the five properties above true simultaneously, and to keep them unfakeable.
