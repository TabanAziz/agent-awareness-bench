# ADR 0001: Python for the harness, over Java and TypeScript

Date: 2026-08-23
Status: accepted

## Context

I work in Java and TypeScript day to day; Python is not my home turf. The harness is a deterministic evaluation driver: it owns a virtual clock, an append-only event log, a tool layer that can lie plausibly, adapters to several model vendors, and scoring that reads only logged events. Whatever language I pick has to make those things boring, reproducible in CI on a machine I do not control, and approachable for outside contributors who want to add probes.

## Decision

Python 3.12+ with `uv` for dependency and interpreter management, `pydantic` v2 for schemas, `pytest` for tests, `ruff` plus `mypy --strict` enforced in CI, and JSONL for traces.

(The floor is 3.12 rather than 3.11 for one concrete reason: the event log's payload contract needs a recursive type alias that both pydantic and mypy resolve natively, and that is exactly PEP 695's `type` statement. Every pre-3.12 encoding of it fails one toolchain or the other.)

## Why

The binding constraint is adapter availability, not language quality. Every model vendor ships a Python SDK first and treats it as the reference client; token accounting conventions, pricing tables, and the existing body of eval tooling all land in Python before anywhere else. A benchmark that must run the same probe against at least three backends lives or dies on how thin its adapters can stay, and thin adapters are the ones that track vendor SDKs.

The second constraint is contributors. The people who write agent evaluations read Python fluently even when they ship other languages at work. A Java harness would shrink the pool of people who can review a probe PR to nearly zero, and this project's legitimacy story depends on outside reviewers actually reading probe code.

Determinism was not a deciding factor because it is a property of discipline, not runtime: integer-valued virtual time, seeded generators, temperature zero, and a CI check that diffs two event logs. Any of the three languages can do that.

Against Java specifically: it is where I am fastest, but the JVM buys nothing here, no ecosystem advantage for evals, heavier dependency machinery for a single-binary CLI, and weaker ergonomics for schema-first JSONL work than pydantic gives me for free.

TypeScript was the real contender: comparable SDK coverage and a type system I trust more than most Python code's. It loses on tooling gravity, tokenizer libraries, cost tables, and the conventions every prior benchmark in [related-work.md](../related-work.md) established are Python-shaped. Probe execution is in-process against virtual filesystems, so it does not require a container runtime.

## Mitigations

Python's weaknesses are exactly the ones that matter here, so they get structural answers rather than good intentions: `mypy --strict` across `src/` in CI, `ruff` with a pinned line length, pydantic models as the single source of truth for every serialized shape, and no duck-typed boundaries between modules. If a boundary accepts "anything dict-like", that is a bug, not flexibility.

## Consequences

CI pins the interpreter through `uv`, so a fresh clone reproduces the environment from the lockfile. Vendor SDK updates will occasionally force adapter churn; that cost is accepted and belongs in the adapter layer only.
