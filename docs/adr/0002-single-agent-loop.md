# ADR 0002: One tested agent loop for every model run

Date: 2026-08-26
Status: accepted

## Context

The invalid M4 pilot used a separate loop that built its request once and never
included subsequent tool results. The tested `AgentLoop` rebuilt messages each
cycle, but the pilot bypassed it. Published measurements therefore described a
path that CI did not exercise.

## Decision

Every model run uses `awarebench.harness.loop.AgentLoop` and a
`ModelAdapter`. Scripts and adapters may configure a run, but they must not
implement another request and tool-use loop.

## Consequences

The CLI is the supported entry point for both stub and vendor-backed runs.
Regression tests require messages sent on later cycles to include earlier tool
results. Any new model integration belongs in `src/awarebench/adapters/` and
is exercised through the same loop.
