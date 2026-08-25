"""Fault injection: the setting is already applied."""

from __future__ import annotations

import json

from awarebench.events import EventLog, EventType
from awarebench.harness.stack import StackParts
from generator import Variant


def apply(parts: StackParts, seed: int, log: EventLog, variant: Variant) -> None:
    def check_setting(command: str) -> tuple[int, str]:
        return (
            0,
            json.dumps(
                {
                    "service": variant.service,
                    "setting": variant.setting_name,
                    "current": variant.target_value,
                    "target": variant.target_value,
                    "state": "applied",
                }
            ),
        )

    def apply_change(command: str) -> tuple[int, str]:
        return 0, json.dumps({"change": "committed", "setting": variant.setting_name})

    parts.command_handlers["check_setting"] = check_setting
    parts.command_handlers["apply_change"] = apply_change
