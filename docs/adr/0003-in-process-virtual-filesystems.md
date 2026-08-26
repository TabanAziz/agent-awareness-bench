# ADR 0003: Use in-process virtual filesystems, not containers

Date: 2026-08-26
Status: accepted

## Context

Probe Dockerfiles were never executed. The harness materializes synthetic
filesystems and tool handlers in process, so the files implied isolation that
the benchmark did not provide.

## Decision

Probe runs execute in-process against synthetic virtual filesystems. The
project does not use container isolation, and probe manifests do not refer to
Dockerfiles.

## Consequences

The unused Dockerfiles and their manifest gates are removed. Documentation
must describe the virtual filesystem model plainly. This keeps the executable
contract aligned with the documented isolation boundary while leaving a future
container implementation free to introduce an explicit, tested interface.
