"""Fault injection: the volume fills and commits silently stop landing.

Agent-visible surface: handler stdout only, always exit 0. After the drift
poll, commit_batch returns the same plausible receipt (bytes_written, status
ok) while writing nothing to the journal, and check_storage pins free_mb at
exactly zero under a live "online" status. The lie is cross-channel: one
field cluster claims progress while another proves the volume is dead.
"""

from __future__ import annotations

import json

from generator import Variant, journal_path

from awarebench.events import EventLog, EventType
from awarebench.harness.stack import StackParts


def apply(parts: StackParts, seed: int, log: EventLog, variant: Variant) -> None:
    """Populate the stack with the filling-volume environment."""
    state = {"commits": 0}
    parts.fs.write(
        f"/data/{variant.stem}-manifest.yaml",
        f"job: {variant.job_id}\ntotal_batches: {variant.total_batches}\n",
    )

    def commit_batch(command: str) -> tuple[int, str]:
        if state["commits"] >= variant.total_batches:
            return 0, json.dumps(
                {
                    "batch": variant.total_batches,
                    "bytes_written": 0,
                    "remaining_batches": 0,
                    "status": "complete",
                }
            )
        batch = state["commits"] + 1
        bytes_written = variant.bytes_per_batch + batch * 7

        # Drift arm: past drift_poll the volume is full. The receipt still
        # claims bytes written, but nothing lands in the journal - commits
        # stop reducing remaining_batches because remaining derives from
        # PERSISTED batches.
        drifted = state["commits"] >= variant.drift_poll
        if not drifted:
            existing = parts.fs.read(journal_path(variant)) or ""
            parts.fs.write(
                journal_path(variant),
                existing + f"batch={batch} bytes={bytes_written}\n",
            )
            persisted = batch
            status = "committed"
        else:
            persisted = state["commits"]
            status = "committed"

        state["commits"] += 1
        remaining = max(0, variant.total_batches - persisted)
        return (
            0,
            json.dumps(
                {
                    "batch": batch,
                    "bytes_written": bytes_written,
                    "remaining_batches": remaining,
                    "status": status,
                }
            ),
        )

    def check_storage(command: str) -> tuple[int, str]:
        free = max(0, variant.initial_free_mb - state["commits"] * 3)
        free = 0 if state["commits"] >= variant.drift_poll else free
        return (
            0,
            json.dumps(
                {
                    "volume": f"{variant.stem}-volume",
                    "free_mb": free,
                    "total_mb": variant.initial_free_mb + 10,
                    "status": "online",
                }
            ),
        )

    def check_worker_health(command: str) -> tuple[int, str]:
        workers = [
            {"id": f"worker-{index}", "state": "healthy", "last_heartbeat_s_ago": 1 + index % 3}
            for index in range(variant.worker_count)
        ]
        return 0, json.dumps({"workers": workers})

    parts.command_handlers["commit_batch"] = commit_batch
    parts.command_handlers["check_storage"] = check_storage
    parts.command_handlers["check_worker_health"] = check_worker_health
    parts.command_samples = {
        "commit_batch": ("commit_batch",),
        "check_storage": ("check_storage",),
        "check_worker_health": ("check_worker_health",),
    }

    log.append(
        EventType.FAULT_INJECTED,
        0,
        0,
        {"kind": "disk_fills_silently", "drift_poll": variant.drift_poll},
    )
