"""Run reports: one frozen JSON summary per probe run."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from awarebench.harness.loop import LoopOutcome
from awarebench.probes.loader import LoadedProbe


class RunReport(BaseModel):
    """Frozen summary of one probe run, written next to its event log."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    probe_id: str
    model: str
    seed: int
    outcome: str
    report_text: str | None
    cycles_used: int
    prompt_tokens: int
    completion_tokens: int
    tool_calls: int
    wall_us_used: int

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
    model: str,
    seed: int,
    outcome: LoopOutcome,
    budget_snapshot: dict[str, int],
) -> RunReport:
    """Assemble a RunReport from the probe identity, outcome, and budget totals."""
    return RunReport(
        probe_id=probe.manifest.id,
        model=model,
        seed=seed,
        outcome=outcome.status,
        report_text=outcome.report_text,
        cycles_used=outcome.cycles_used,
        prompt_tokens=budget_snapshot["prompt_tokens"],
        completion_tokens=budget_snapshot["completion_tokens"],
        tool_calls=budget_snapshot["tool_calls"],
        wall_us_used=budget_snapshot["wall_us_used"],
    )
