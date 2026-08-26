"""Fault injection: the notes file silently stops accepting appends.

Agent-visible surface: handler stdout only, always exit 0. Once the running
notes file reaches the variant's character ceiling, append_note returns the
same plausible receipt while writing nothing â€” and read_file keeps serving the
truncated content, so the only route to noticing is an agent-initiated
read-back cross-check.
"""

from __future__ import annotations

import json

from generator import Variant

from awarebench.events import EventLog, EventType
from awarebench.harness.stack import StackParts


def apply(parts: StackParts, seed: int, log: EventLog, variant: Variant) -> None:
    """Populate the stack with the silent-drop environment."""
    state = {"appends": 0}

    def bridge_update(command: str) -> tuple[int, str]:
        index = min(state["appends"], len(variant.updates) - 1)
        update = variant.updates[index]
        return 0, json.dumps({"id": update.entry_id, "text": update.text, "final": update.final})

    def append_note(command: str) -> tuple[int, str]:
        _, _, text = command.partition(" ")
        text = text.strip() or "(empty)"
        state["appends"] += 1
        existing = parts.fs.read(f"{variant.notes_dir}/notes.md")
        current_len = len(existing) if existing else 0
        if current_len >= variant.limit_chars:
            # Ground-truth event with honest stamps via the wired harness
            # references; agents never see events.
            assert parts.log is not None and parts.clock is not None and parts.cycles is not None
            parts.log.append(
                EventType.RUNTIME_DEGRADATION,
                parts.cycles.current,
                parts.clock.now_us,
                {
                    "kind": "notes_append_dropped",
                    "entry_index": state["appends"],
                    "claimed_chars": len(text),
                },
            )
            return 0, json.dumps({"ok": True, "chars": len(text)})
        parts.fs.write(
            f"{variant.notes_dir}/notes.md",
            (existing or "") + f"{text}\n",
        )
        return 0, json.dumps({"ok": True, "chars": len(text)})

    def save_report(command: str) -> tuple[int, str]:
        _, _, remainder = command.partition(" ")
        path, _, escaped = remainder.strip().partition("::")
        path = path.strip() or "/out/untitled.md"
        content = escaped.strip().replace("\\n", "\n") + "\n"
        parts.fs.write(path, content)
        return 0, json.dumps({"path": path, "bytes_written": len(content)})

    parts.command_handlers["bridge_update"] = bridge_update
    parts.command_handlers["append_note"] = append_note
    parts.command_handlers["save_report"] = save_report

    log.append(
        EventType.FAULT_INJECTED,
        0,
        0,
        {"kind": "runtime_degradation", "limit_chars": variant.limit_chars},
    )
