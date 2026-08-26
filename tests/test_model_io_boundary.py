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
    dynamic_import_names = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"builtins", "importlib"}:
            for alias in node.names:
                if alias.name in {"__import__", "import_module"}:
                    dynamic_import_names.add(alias.asname or alias.name)

    def literal_string(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = literal_string(node.left)
            right = literal_string(node.right)
            return left + right if left is not None and right is not None else None
        return None

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
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and (func.value.id, func.attr) in {("os", "system"), ("os", "popen")}
            ):
                violations.append(
                    f"{filename}:{node.lineno}: process transport {func.value.id}.{func.attr}"
                )
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "complete"
                and not filename.replace("\\", "/").endswith("/harness/loop.py")
            ):
                violations.append(f"{filename}:{node.lineno}: adapter completion outside AgentLoop")
            elif (isinstance(func, ast.Name) and func.id in dynamic_import_names) or (
                isinstance(func, ast.Attribute) and func.attr in {"__import__", "import_module"}
            ):
                violations.append(f"{filename}:{node.lineno}: dynamic import")
            elif (
                isinstance(func, ast.Name)
                and func.id == "getattr"
                and len(node.args) >= 2
                and literal_string(node.args[1]) == "complete"
            ):
                violations.append(f"{filename}:{node.lineno}: dynamic adapter completion")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "complete"
            and not filename.replace("\\", "/").endswith("/harness/loop.py")
        ):
            violations.append(
                f"{filename}:{node.lineno}: adapter completion reference outside AgentLoop"
            )
    return violations


def _is_adapter_file(path: Path) -> bool:
    return path.is_relative_to(Path("src/awarebench/adapters"))


def _production_python_files() -> list[Path]:
    paths = [
        *Path("src").rglob("*.py"),
        *Path("tools").rglob("*.py"),
        *Path("probes").rglob("*.py"),
        *Path(".").glob("*.py"),
    ]
    return sorted(path for path in paths if not _is_adapter_file(path))


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
        '__import__("urllib.request", fromlist=["urlopen"]).urlopen("x")\n',
        'import importlib\nimportlib.import_module("urllib" + ".request")\n',
        'from importlib import import_module as load\nload("urllib" + ".request")\n',
        'import builtins\nbuiltins.__import__("urllib" + ".request", fromlist=["urlopen"])\n',
        "endpoint = 'https://example.invalid/model'\n",
        "from awarebench.adapters import OpenRouterAdapter\nadapter.complete([])\n",
        "call = adapter.complete\ncall([])\n",
        'getattr(adapter, "complete")([])\n',
        'getattr(adapter, "com" + "plete")([])\n',
    ],
)
def test_boundary_scanner_rejects_known_bypass_shapes(source: str) -> None:
    assert _scan_source(source)


def test_only_the_exact_adapter_package_is_exempt() -> None:
    assert _is_adapter_file(Path("src/awarebench/adapters/openrouter.py"))
    assert not _is_adapter_file(Path("tools/adapters/rogue_runner.py"))
    source = "import urllib.request\nENDPOINT = 'https://evil.example/model'\n"
    assert _scan_source(source, "tools/adapters/rogue_runner.py")
