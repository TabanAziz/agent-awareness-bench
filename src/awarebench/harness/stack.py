"""Assembly of the per-probe environment pieces a ToolHost needs.

Probe artifact modules (generator.py / injection.py / control.py) mutate a
StackParts in place: the generator derives the variant from a seed, and the
injection or control module registers handlers, writes files, and configures
faults. The harness owns the types; probes own their contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from awarebench.harness.tools import CommandHandler, FaultSet, VirtualFilesystem


@dataclass
class StackParts:
    """Mutable bundle of everything a probe environment contributes."""

    fs: VirtualFilesystem = field(default_factory=VirtualFilesystem)
    command_handlers: dict[str, CommandHandler] = field(default_factory=dict)
    http_table: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    faults: FaultSet = field(default_factory=FaultSet)
