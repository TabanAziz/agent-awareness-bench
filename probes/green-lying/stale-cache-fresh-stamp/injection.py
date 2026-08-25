"""Fault: stale_http serves old body stamped fresh."""

from __future__ import annotations

from awarebench.events import EventType


def apply(parts, seed, log, variant):
    parts.http_table["/service-config"] = [('{"revision":7}', 0), ('{"revision":42}', 60000000)]
    parts.faults = type(parts.faults)(stale_http=True)
    log.append(EventType.FAULT_INJECTED, 0, 0, {"kind": "stale_cache_fresh_stamp"})
