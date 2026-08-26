"""Run reports: one frozen JSON summary per probe run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from awarebench.harness.loop import LoopOutcome
from awarebench.probes.loader import LoadedProbe


class RunReport(BaseModel):
    """Frozen summary of one probe run, written next to its event log."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = 2
    probe_id: str
    model: str
    backend: str
    requested_model: str | None
    variant: Literal["fault", "control"]
    seed: int
    outcome: str
    report_text: str | None
    cycles_used: int
    prompt_tokens: int
    completion_tokens: int
    tool_calls: int
    wall_us_used: int
    # Predicate results over this run's event log; empty until scored. `passed`
    # is None when no predicate set was evaluated, else the AND of all results.
    predicates: dict[str, bool] = {}
    passed: bool | None = None

    def write_json(self, path: str | Path) -> None:
        """Serialize the report and write it, creating parent directories."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            allow_nan=False,
            ensure_ascii=True,
        )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")


def build_report(
    probe: LoadedProbe,
    *,
    backend: str,
    requested_model: str | None,
    variant: Literal["fault", "control"],
    seed: int,
    outcome: LoopOutcome,
    budget_snapshot: dict[str, int],
    predicates: dict[str, bool] | None = None,
) -> RunReport:
    """Assemble a RunReport from the probe identity, outcome, and budget totals.

    When a predicate-result mapping is supplied it is recorded verbatim and
    `passed` becomes the AND of its values; without one the run is unscored
    (`predicates` empty, `passed` None).
    """
    scored = predicates if predicates is not None else {}
    passed: bool | None = all(scored.values()) if predicates is not None else None
    return RunReport(
        probe_id=probe.manifest.id,
        model=backend if requested_model is None else f"{backend}:{requested_model}",
        backend=backend,
        requested_model=requested_model,
        variant=variant,
        seed=seed,
        outcome=outcome.status,
        report_text=outcome.report_text,
        cycles_used=outcome.cycles_used,
        prompt_tokens=budget_snapshot["prompt_tokens"],
        completion_tokens=budget_snapshot["completion_tokens"],
        tool_calls=budget_snapshot["tool_calls"],
        wall_us_used=budget_snapshot["wall_us_used"],
        predicates=scored,
        passed=passed,
    )
