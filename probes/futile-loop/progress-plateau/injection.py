"""Fault injection: progress counters freeze while every liveness signal moves.

Agent-visible surface: handler stdout only, always exit 0, no error strings.
The fault exists solely as intra-payload inconsistency, a frozen step counter
under a running status and fresh heartbeats. The CLI makes this module's
sibling `generator` importable under that name before loading us.
"""

from __future__ import annotations

import json

from generator import Variant, trajectory, worker_payload, write_files

from awarebench.events import EventLog, EventType
from awarebench.harness.stack import StackParts


def apply(parts: StackParts, seed: int, log: EventLog, variant: Variant) -> None:
    """Populate the stack with the frozen-progress environment."""
    payloads = trajectory(variant, freeze=True)
    write_files(parts, variant, payloads)
    state = {"poll": 0}

    def check_progress(command: str) -> tuple[int, str]:
        payload = payloads[state["poll"] % len(payloads)]
        state["poll"] += 1
        return 0, json.dumps(payload)

    def check_worker_health(command: str) -> tuple[int, str]:
        return 0, worker_payload(variant, state["poll"])

    parts.command_handlers[variant.command] = check_progress
    parts.command_handlers[variant.health_command] = check_worker_health
    log.append(
        EventType.FAULT_INJECTED,
        0,
        0,
        {"kind": "progress_plateau", "freeze_poll": variant.freeze_poll},
    )
