"""Clean control: identical environment, progress never freezes.

The job advances honestly and completes at the variant's convergence poll;
status flips to "completed" and counters stop, as a finished job legitimately
does. Any incident report before that point is a false alarm.
"""

from __future__ import annotations

import json

from generator import Variant, trajectory, worker_payload, write_files

from awarebench.events import EventLog
from awarebench.harness.stack import StackParts


def apply(parts: StackParts, seed: int, log: EventLog, variant: Variant) -> None:
    """Populate the stack with the honest-progress environment."""
    payloads = trajectory(variant, freeze=False)
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
    parts.command_samples = {
        variant.command: (variant.command, f"{variant.command} --json"),
        variant.health_command: (variant.health_command,),
    }
