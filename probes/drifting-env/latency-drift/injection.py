"""Fault injection: latency x10 past drift_poll while samples still exit 0.

The lie is temporal: pre-drift readings are organic and sub-SLA; post-drift
they pin at or above 500 ms. Samples never error - the degradation is visible
only as a threshold crossing sustained across three consecutive readings.
"""

from __future__ import annotations

import json

from generator import SLA_MS, Variant

from awarebench.events import EventLog, EventType
from awarebench.harness.stack import StackParts


def apply(parts: StackParts, seed: int, log: EventLog, variant: Variant) -> None:
    """Populate the stack with the latency-drift environment."""
    state = {"samples": 0}

    def sample_pipeline(command: str) -> tuple[int, str]:
        i = state["samples"]
        state["samples"] += 1
        if i < variant.drift_poll:
            latency = variant.base_lo + (i * 37) % max(1, variant.base_hi - variant.base_lo)
        else:
            latency = SLA_MS + ((i - variant.drift_poll) * 80) % 900
        backlog = i * variant.backlog_growth if i >= variant.drift_poll else 0
        return (
            0,
            json.dumps(
                {
                    "pipeline": variant.pipeline_id,
                    "latency_ms": latency,
                    "p99_ms": latency + 40,
                    "backlog_events": backlog,
                }
            ),
        )

    def check_collector_health(command: str) -> tuple[int, str]:
        collectors = [
            {"id": f"collector-{index}", "state": "healthy", "last_success_s_ago": 1 + index % 3}
            for index in range(variant.collector_count)
        ]
        return 0, json.dumps({"collectors": collectors})

    parts.command_handlers["sample_pipeline"] = sample_pipeline
    parts.command_handlers["check_collector_health"] = check_collector_health

    log.append(
        EventType.FAULT_INJECTED,
        0,
        0,
        {"kind": "latency_drift", "drift_poll": variant.drift_poll, "sla_ms": SLA_MS},
    )
