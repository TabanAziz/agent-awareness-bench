# Invalid M4 Pilot Postmortem

## What I published

I published M4 pilot results for `futile-loop/progress-plateau` and described them as real model measurements. The results document reported detection rates and Action Gap values, then concluded that the say-do gap was maximal. Those claims were not supported by the code path that produced the runs, so I have removed the document and retract every figure and conclusion in it.

## Why it was wrong

The pilot runner duplicated the agent loop instead of using `AgentLoop`. It built the request payload once before entering the cycle loop:

```python
messages = [{"role": "system", "content": system_msg}]
for role, content in context.messages():
    messages.append({"role": role, "content": content})

for cycle in range(max_cycles):
    text, pt, ct = _openrouter_call(api_key, model, messages, max_tokens=2048)
    context.add("assistant", text)
    context.add("user", result_text)
```

`context.add(...)` updated the `ContextWindow`, but the runner never rebuilt `messages` from that context. This fixed payload gave every request the same initial prompt. The models did not receive prior responses or tool results and could not observe the frozen counter the probe was meant to test.

The retained raw run data made the defect visible. The two completed seeds each recorded 5,980 prompt tokens over 20 cycles, exactly 299 prompt tokens per call. The final completed call in the aborted third seed also recorded 299 prompt tokens. Tool use changed between runs, but prompt size did not grow with the transcript.

The Action Gap claims were independently invalid. The metric is defined as the probability of naming the problem minus the probability of changing behavior correctly within the configured window. The repository has no judge layer that can determine whether reasoning or a report named the actual problem. I therefore could not compute the first input. Even if the reported detection rate of zero had been valid, subtracting zero behavior change from zero detection would give zero, not the published value of 1.0.

The interpretation also described a different probe. The text about latency crossing an SLA and three consecutive breaches belongs to `drifting-env/latency-drift`, not `futile-loop/progress-plateau`.

## How I found it

I compared the pilot runner with the tested harness loop. `AgentLoop` rebuilds its messages inside each cycle; the pilot runner did not. I then checked the retained token counts and confirmed the constant 299-token request pattern.

The missing semantic judge was not a new discovery. The PR 20 red-team review had already recorded that there was no report-content machine check and that the say-do gap was invisible without a judge. That limitation was accepted for the behavioral probe set and deferred to a later judge milestone. I then failed to preserve that boundary when publishing the pilot.

## What changed

I removed the invalid results instead of replacing them with an interpretation of broken runs. The README now states that there are no valid pilot results, the judge layer is not implemented, and Dockerfiles are not executed.

I am also applying one publication rule to the repair work: no number reaches a committed document unless the same code path is exercised by CI. Model runs must go through `AgentLoop` and a `ModelAdapter`; figure-producing commands must work from a clean checkout and be covered by tests; documents must name the exact command; and an uncomputable metric must be described as uncomputable rather than printed with a placeholder value. Until those controls and the judge layer are implemented and tested, the benchmark has no valid pilot result.

## Process failure

The review process separated a known limitation from the claims that depended on it. Accepting the absent judge as a deferred limitation was reasonable only while the repository made behavioral claims. The later pilot treated zero reports as making content grading irrelevant, then published an Action Gap value anyway. At the same time, the pilot runner bypassed the tested harness path, and the results document was reviewed without checking that its narrative matched the named probe.

The failure was not that the review trail lacked the warning. The warning existed. I allowed a later milestone to publish through an untested duplicate path and to promote a known missing input into a measured headline metric.
