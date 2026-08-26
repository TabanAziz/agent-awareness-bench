"""Executable boundary: tool scripts never own model transports."""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_MODULES = (
    "aiohttp",
    "anthropic",
    "httpx",
    "openai",
    "requests",
    "urllib.request",
)


def _is_forbidden_module(module: str) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_MODULES)


def test_tools_do_not_import_http_clients_or_embed_http_endpoints() -> None:
    violations: list[str] = []
    for path in sorted(Path("tools").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden_module(alias.name):
                        violations.append(f"{path.as_posix()}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _is_forbidden_module(module):
                    violations.append(f"{path.as_posix()}:{node.lineno}: from {module}")
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and ("http://" in node.value or "https://" in node.value)
            ):
                violations.append(f"{path.as_posix()}:{node.lineno}: HTTP endpoint literal")
    assert violations == []
