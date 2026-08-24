"""Global staleness guard for the harness package export surface."""

from __future__ import annotations

import importlib

import awarebench.harness as harness_package


def test_every_exported_name_resolves_to_its_source_symbol() -> None:
    assert len(harness_package.__all__) == len(set(harness_package.__all__))
    for name in harness_package.__all__:
        symbol = getattr(harness_package, name)
        source_module_name = getattr(symbol, "__module__", None)
        if source_module_name is not None:
            source = importlib.import_module(source_module_name)
            assert getattr(source, name) is symbol, f"stale export: {name}"
