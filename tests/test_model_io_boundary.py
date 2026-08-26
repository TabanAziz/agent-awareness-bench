"""Executable boundary: only adapters own model transports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_FORBIDDEN_MODULES = (
    "aiohttp",
    "anthropic",
    "http.client",
    "httpx",
    "openai",
    "requests",
    "socket",
    "subprocess",
    "urllib",
)
_FORBIDDEN_SYMBOLS = {"os.popen", "os.system"}


def _is_forbidden_module(module: str) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_MODULES)


def _scan_source(source: str, filename: str = "candidate.py") -> list[str]:
    violations: list[str] = []
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    violations.append(f"{filename}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            qualified = [f"{module}.{alias.name}" if module else alias.name for alias in node.names]
            if (
                _is_forbidden_module(module)
                or any(_is_forbidden_module(name) for name in qualified)
                or any(name in _FORBIDDEN_SYMBOLS for name in qualified)
            ):
                violations.append(f"{filename}:{node.lineno}: from {module}")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and ("http://" in node.value or "https://" in node.value)
        ):
            violations.append(f"{filename}:{node.lineno}: HTTP endpoint literal")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id, node.func.attr) in {("os", "system"), ("os", "popen")}
        ):
            violations.append(
                f"{filename}:{node.lineno}: process transport {node.func.value.id}.{node.func.attr}"
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "complete"
            and not filename.replace("\\", "/").endswith("/harness/loop.py")
        ):
            violations.append(f"{filename}:{node.lineno}: adapter completion outside AgentLoop")
    return violations


def _production_python_files() -> list[Path]:
    paths = [
        *Path("src").rglob("*.py"),
        *Path("tools").rglob("*.py"),
        *Path(".").glob("*.py"),
    ]
    return sorted(path for path in paths if "adapters" not in path.parts)


def test_only_adapters_import_http_clients_or_embed_http_endpoints() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        violations.extend(_scan_source(path.read_text(encoding="utf-8"), path.as_posix()))
    assert violations == []


@pytest.mark.parametrize(
    "source",
    [
        "from urllib import request\nrequest.urlopen('x')\n",
        "import http.client\n",
        (
            "import subprocess\nendpoint = 'https' + '://example.invalid'\n"
            "subprocess.run(['curl', endpoint])\n"
        ),
        "import socket\n",
        "from os import system\nsystem('curl example.invalid')\n",
        "import os\nos.system('curl https' + '://example.invalid')\n",
        "endpoint = 'https://example.invalid/model'\n",
        "from awarebench.adapters import OpenRouterAdapter\nadapter.complete([])\n",
    ],
)
def test_boundary_scanner_rejects_known_bypass_shapes(source: str) -> None:
    assert _scan_source(source)
