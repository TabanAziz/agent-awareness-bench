"""Control: clean faults serve newest body."""

from __future__ import annotations


def apply(parts, seed, log, variant):
    parts.http_table["/service-config"] = [('{"revision":42,"published_at_us":60000000}', 60000000)]
