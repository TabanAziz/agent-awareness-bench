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
    # (role, content) pairs the CLI seeds into the context window before the
    # loop starts; the only way a killable constraint lives inside the
    # compactable transcript (system task text is rebuilt every cycle).
    seed_messages: list[tuple[str, str]] = field(default_factory=list)
    # Variant-driven window overrides consulted by the CLI when building the
    # ContextWindow: None falls back to the manifest, then --context-tokens.
    # drop_policy names a registered policy factory ("drop_oldest" or
    # "drop_oldest_half"); None uses the harness default.
    context_max_tokens: int | None = None
    drop_policy: str | None = None
    # Harness references wired in by the CLI before apply() runs, so handler
    # closures can emit ground-truth events with honest stamps. Probes must
    # not mutate these.
    clock: object | None = None
    cycles: object | None = None
    log: object | None = None
