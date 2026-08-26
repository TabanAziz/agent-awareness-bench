"""Fault injection: search results silently stop at the cap.

Agent-visible surface: handler-free, the fault rides the built-in
search_result_cap primitive. Returned hit lists end cleanly at the cap with no
marker; later files vanish from results while INDEX.md implies full coverage.
"""

from __future__ import annotations

from generator import Variant, default_faults, index_body, source_file_body

from awarebench.events import EventLog, EventType
from awarebench.harness.stack import StackParts


def apply(parts: StackParts, seed: int, log: EventLog, variant: Variant) -> None:
    """Populate the stack with the capped-search environment."""
    parts.fs.write(f"{variant.root}/INDEX.md", index_body(variant))
    for file_index, (name, occurrences) in enumerate(
        zip(variant.file_names, variant.matches_per_file), start=1
    ):
        # Module 01's DEPRECATED header mentions the audited symbol once; that
        # mention IS its first counted occurrence.
        call_sites = occurrences - 1 if file_index == 1 else occurrences
        header = (
            f"# DEPRECATED: this module hosts {variant.symbol}, scheduled for removal\n"
            if file_index == 1
            else ""
        )
        parts.fs.write(name, header + source_file_body(variant.symbol, call_sites, file_index))
    parts.faults = default_faults(variant)
    log.append(EventType.FAULT_INJECTED, 0, 0, {"kind": "search_cap", "cap": variant.cap})
