# Contributing

## Setup

```bash
git clone https://github.com/TabanAziz/agent-awareness-bench.git
cd agent-awareness-bench
uv sync
uv run pytest -q
```

Python 3.12+ is required (PEP 695 recursive type aliases).

## Adding a probe

1. Open an issue describing the fault, the detectability point, and the success criteria.
2. Create `probes/<class>/<probe-id>/` with `probe.yaml`, `generator.py`, `injection.py`, `control.py`, and a maintainer-facing `README.md`.
3. Run `uv run python tools/leakage_scan.py .` to verify zero eval signals.
4. Write end-to-end tests showing compliant behavior passes and blind continuation fails.
5. Open a PR; the loader must accept your manifest and all gates must pass.

Every probe must have:
- A machine-checkable detectability point (registered predicate).
- A clean control variant that differs only by removing the fault.
- A procedural generator producing at least 20 seed-distinct variants.
- A cheat surface analysis in probe.yaml.
- Zero error signals anywhere agent-visible.

## Code style

Ruff format + ruff check + mypy --strict are enforced. All boundaries use pydantic models or typed aliases. No duck-typed parameters at module edges.

## Commit messages

Short, lowercase, imperative, one line. Example: `add disk-fills probe with cross-channel lie`.
