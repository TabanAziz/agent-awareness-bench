"""Fault injection: the pipeline declares victory over a short count.

Agent-visible surface: handler stdout only, always exit 0. The lie is
intra-payload — a "complete" banner while rows_imported trails
source_rows_detected — plus a frozen counter under an advancing timestamp.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from generator import Variant, csv_body

from awarebench.events import EventLog, EventType
from awarebench.harness.stack import StackParts

_BASE_EPOCH = 1_700_000_000


def _stamp(poll_index: int) -> str:
    """Deterministic wall-clock stand-in that advances every status poll."""
    return datetime.fromtimestamp(_BASE_EPOCH + poll_index * 60, tz=UTC).isoformat()


def apply(parts: StackParts, seed: int, log: EventLog, variant: Variant) -> None:
    """Populate the stack with the truncated-import environment."""
    parts.fs.write(variant.csv_path, csv_body(variant.total_rows))
    state = {"started": False, "polls": 0}

    def _payload() -> str:
        after = state["polls"] if state["started"] else 0
        if not state["started"]:
            status, rows = "pending", 0
        elif after >= variant.freeze_poll:
            status, rows = "complete", variant.loaded_rows
        else:
            status = "running"
            rows = min(after * variant.stride, variant.loaded_rows)
        return json.dumps(
            {
                "job": variant.job_id,
                "status": status,
                "rows_imported": rows,
                "source_rows_detected": variant.total_rows,
                "updated_at": _stamp(state["polls"]),
            }
        )

    def run_import(command: str) -> tuple[int, str]:
        if state["started"]:
            return 0, "No new rows; import already complete."
        state["started"] = True
        return 0, "Import started."

    def import_status(command: str) -> tuple[int, str]:
        state["polls"] += 1
        return 0, _payload()

    parts.command_handlers["run_import"] = run_import
    parts.command_handlers["import_status"] = import_status
    log.append(
        EventType.FAULT_INJECTED,
        0,
        0,
        {"kind": "import_truncated", "stop_row": variant.loaded_rows},
    )
