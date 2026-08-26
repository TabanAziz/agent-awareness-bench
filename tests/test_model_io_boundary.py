"""Executable boundary: only adapters own model transports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_FORBIDDEN_MODULES = (
    "aiohttp",
    "anyio",
    "anthropic",
    "asyncio",
    "builtins",
    "ftplib",
    "gc",
    "grpc",
    "http.client",
    "httpx",
    "inspect",
    "openai",
    "operator",
    "requests",
    "socket",
    "smtplib",
    "subprocess",
    "telnetlib",
    "trio",
    "urllib",
    "urllib3",
    "websocket",
    "websockets",
    "xmlrpc",
)
_FORBIDDEN_SYMBOLS = {"os.popen", "os.system", "types.FunctionType"}
_FORBIDDEN_REFLECTION_ATTRIBUTES = {
    "__dict__",
    "__getattribute__",
    "__globals__",
    "__subclasses__",
    "FunctionType",
    "_getframe",
    "f_builtins",
}
_ADAPTER_CLASS_NAMES = {
    "AnthropicAdapter",
    "OpenAIAdapter",
    "OpenRouterAdapter",
    "StubAdapter",
}
_ADAPTER_PRIVATE_TRANSPORT_ATTRIBUTES = {
    "_client",
    "_default_client",
    "_default_transport",
    "_ensure_client",
    "_import_anthropic_sdk",
    "_import_openai_sdk",
    "_transport",
}
_FORBIDDEN_DYNAMIC_ADAPTER_ATTRIBUTES = (
    {"complete", "complete_model"} | _ADAPTER_CLASS_NAMES | _ADAPTER_PRIVATE_TRANSPORT_ATTRIBUTES
)
_ALLOWED_NON_ADAPTER_IMPORT_ROOTS = {
    "__future__",
    "argparse",
    "awarebench",
    "collections",
    "copy",
    "dataclasses",
    "datetime",
    "generator",
    "hashlib",
    "importlib",
    "json",
    "math",
    "pathlib",
    "pydantic",
    "random",
    "re",
    "shlex",
    "statistics",
    "sys",
    "traceback",
    "types",
    "typing",
    "yaml",
}
_ALLOWED_COMPLETION_FILES: set[str] = set()
_ALLOWED_GATEWAY_FILES = {
    "src/awarebench/harness/loop.py",
    "src/awarebench/scoring/judge.py",
}
_ALLOWED_ADAPTER_IMPORTS = {
    "src/awarebench/cli.py": {
        "AnthropicAdapter",
        "ModelAdapter",
        "OpenAIAdapter",
        "OpenRouterAdapter",
        "StubAdapter",
    },
    "tools/solvability_check.py": {"StubAdapter"},
    "src/awarebench/solvability.py": {"StubAdapter"},
    "src/awarebench/harness/context.py": {"message_token_text"},
    "src/awarebench/harness/loop.py": {
        "AdapterError",
        "AdapterMessage",
        "ModelAdapter",
        "complete_model",
    },
    "src/awarebench/scoring/judge.py": {
        "AdapterError",
        "AdapterMessage",
        "AdapterResponse",
        "ModelAdapter",
        "complete_model",
    },
    "src/awarebench/scoring/judge_validation.py": {"AdapterMessage", "AdapterResponse"},
}
_FILE_SCOPED_IMPORT_ROOTS = {
    "importlib": {"src/awarebench/cli.py"},
    "types": {"src/awarebench/cli.py"},
}


def _is_forbidden_module(module: str) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_MODULES)


def _completion_allowed(filename: str) -> bool:
    normalized = filename.replace("\\", "/")
    return any(normalized.endswith(path) for path in _ALLOWED_COMPLETION_FILES)


def _gateway_allowed(filename: str) -> bool:
    normalized = filename.replace("\\", "/")
    return any(normalized.endswith(path) for path in _ALLOWED_GATEWAY_FILES)


def _adapter_import_allowed(filename: str, imported_names: set[str]) -> bool:
    normalized = filename.replace("\\", "/")
    return any(
        normalized.endswith(path) and imported_names <= allowed_names
        for path, allowed_names in _ALLOWED_ADAPTER_IMPORTS.items()
    )


def _import_root_allowed(filename: str, root: str) -> bool:
    if root not in _ALLOWED_NON_ADAPTER_IMPORT_ROOTS:
        return False
    scoped_paths = _FILE_SCOPED_IMPORT_ROOTS.get(root)
    if scoped_paths is None:
        return True
    normalized = filename.replace("\\", "/")
    return any(normalized.endswith(path) for path in scoped_paths)


def _scan_source(source: str, filename: str = "candidate.py") -> list[str]:
    violations: list[str] = []
    tree = ast.parse(source, filename=filename)

    def literal_string(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = literal_string(node.left)
            right = literal_string(node.right)
            return left + right if left is not None and right is not None else None
        if isinstance(node, ast.JoinedStr):
            joined_parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    joined_parts.append(value.value)
                elif (
                    isinstance(value, ast.FormattedValue)
                    and value.conversion == -1
                    and value.format_spec is None
                    and (part := literal_string(value.value)) is not None
                ):
                    joined_parts.append(part)
                else:
                    return None
            return "".join(joined_parts)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and len(node.args) == 1
            and not node.keywords
        ):
            separator = literal_string(node.func.value)
            values = node.args[0]
            if separator is None or not isinstance(values, (ast.List, ast.Tuple)):
                return None
            joined_values = [literal_string(element) for element in values.elts]
            if any(part is None for part in joined_values):
                return None
            return separator.join(part for part in joined_values if part is not None)
        return None

    dynamic_import_names = {"__import__"}
    gateway_names = {"complete_model"}
    string_constants: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"builtins", "importlib"}:
            for alias in node.names:
                if alias.name in {"__import__", "import_module"}:
                    dynamic_import_names.add(alias.asname or alias.name)
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "complete_model":
                    gateway_names.add(alias.asname or alias.name)
        if isinstance(node, ast.Assign):
            value = literal_string(node.value)
            if value is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        string_constants[target.id] = value

    def resolved_string(node: ast.AST) -> str | None:
        literal = literal_string(node)
        if literal is not None:
            return literal
        return string_constants.get(node.id) if isinstance(node, ast.Name) else None

    def is_safe_dynamic_getattr(call: ast.Call) -> bool:
        if len(call.args) < 2:
            return False
        target, attribute = call.args[:2]
        normalized = filename.replace("\\", "/")
        return (
            normalized.endswith("src/awarebench/cli.py")
            and isinstance(target, ast.Name)
            and target.id == "args"
            and isinstance(attribute, ast.Name)
            and attribute.id == "field"
        ) or (
            normalized.endswith("src/awarebench/probes/loader.py")
            and isinstance(target, ast.Name)
            and target.id == "manifest"
            and isinstance(attribute, ast.Name)
            and attribute.id == "field_name"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                if not _import_root_allowed(filename, root):
                    violations.append(f"{filename}:{node.lineno}: unapproved import {alias.name}")
                if alias.name == "awarebench.adapters" or alias.name.startswith(
                    "awarebench.adapters."
                ):
                    violations.append(f"{filename}:{node.lineno}: unapproved adapter import")
                if _is_forbidden_module(alias.name):
                    violations.append(f"{filename}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", maxsplit=1)[0]
            if not _import_root_allowed(filename, root):
                violations.append(f"{filename}:{node.lineno}: unapproved import from {module}")
            qualified = [f"{module}.{alias.name}" if module else alias.name for alias in node.names]
            imports_adapter_parent = module == "awarebench" and any(
                alias.name == "adapters" for alias in node.names
            )
            if imports_adapter_parent or (
                module.startswith("awarebench.adapters")
                and not _adapter_import_allowed(filename, {alias.name for alias in node.names})
            ):
                violations.append(f"{filename}:{node.lineno}: unapproved adapter import")
            if not module.startswith("awarebench.adapters") and any(
                alias.name in _ADAPTER_CLASS_NAMES for alias in node.names
            ):
                violations.append(f"{filename}:{node.lineno}: adapter class re-export import")
            if (
                _is_forbidden_module(module)
                or any(_is_forbidden_module(name) for name in qualified)
                or any(name in _FORBIDDEN_SYMBOLS for name in qualified)
            ):
                violations.append(f"{filename}:{node.lineno}: from {module}")
            if any(alias.name == "complete_model" for alias in node.names) and not _gateway_allowed(
                filename
            ):
                violations.append(f"{filename}:{node.lineno}: completion gateway import")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and ("http://" in node.value or "https://" in node.value)
        ):
            violations.append(f"{filename}:{node.lineno}: HTTP endpoint literal")
        elif isinstance(node, ast.Call):
            func = node.func
            gateway_call = (isinstance(func, ast.Name) and func.id in gateway_names) or (
                isinstance(func, ast.Attribute) and func.attr == "complete_model"
            )
            if gateway_call and not _gateway_allowed(filename):
                violations.append(f"{filename}:{node.lineno}: completion gateway call")
            elif (
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
                and not _completion_allowed(filename)
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
                and resolved_string(node.args[1]) in _FORBIDDEN_DYNAMIC_ADAPTER_ATTRIBUTES
            ) or (
                isinstance(func, ast.Attribute)
                and func.attr == "__getattribute__"
                and any(
                    resolved_string(argument) in _FORBIDDEN_DYNAMIC_ADAPTER_ATTRIBUTES
                    for argument in node.args
                )
            ):
                violations.append(f"{filename}:{node.lineno}: dynamic adapter completion")
            elif (
                isinstance(func, ast.Name)
                and func.id == "getattr"
                and len(node.args) >= 2
                and resolved_string(node.args[1]) is None
                and not is_safe_dynamic_getattr(node)
            ):
                violations.append(f"{filename}:{node.lineno}: unresolved dynamic attribute")
            elif isinstance(func, ast.Attribute) and func.attr == "__getattribute__":
                violations.append(f"{filename}:{node.lineno}: direct dynamic attribute access")
            elif isinstance(func, ast.Subscript) and (
                resolved_string(func.slice) in {"complete", "complete_model"}
                or (
                    isinstance(func.value, ast.Call)
                    and isinstance(func.value.func, ast.Name)
                    and func.value.func.id in {"globals", "locals", "vars"}
                )
            ):
                violations.append(f"{filename}:{node.lineno}: dynamic completion lookup")
            elif (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "sys"
                and func.value.attr == "modules"
            ):
                violations.append(f"{filename}:{node.lineno}: runtime module registry lookup")
            elif isinstance(func, ast.Name) and func.id in {
                "eval",
                "exec",
                "compile",
                "globals",
                "locals",
                "vars",
            }:
                violations.append(
                    f"{filename}:{node.lineno}: dynamic execution or namespace lookup"
                )
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "sys"
            and node.value.attr == "modules"
        ):
            violations.append(f"{filename}:{node.lineno}: runtime module registry lookup")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_REFLECTION_ATTRIBUTES:
            violations.append(f"{filename}:{node.lineno}: dynamic reflection attribute")
        elif isinstance(node, ast.Attribute) and node.attr in _ADAPTER_PRIVATE_TRANSPORT_ATTRIBUTES:
            violations.append(f"{filename}:{node.lineno}: private adapter transport access")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "complete_model"
            and not _gateway_allowed(filename)
        ):
            violations.append(f"{filename}:{node.lineno}: completion gateway reference")
        elif isinstance(node, ast.Attribute) and node.attr in _ADAPTER_CLASS_NAMES:
            violations.append(f"{filename}:{node.lineno}: adapter class re-export reference")
        elif isinstance(node, ast.Name) and node.id == "__builtins__":
            violations.append(f"{filename}:{node.lineno}: builtins namespace access")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "complete"
            and not _completion_allowed(filename)
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
        "import asyncio\nasyncio.run(asyncio.open_connection('127.0.0.1', 9999))\n",
        (
            "import logging.handlers\nimport socketserver\nimport threading\n"
            "logging.handlers.SocketHandler('127.0.0.1', 9999).makeSocket()\n"
        ),
        "import novel_transport\nnovel_transport.connect('model.example')\n",
        "import importlib.util\nimportlib.util.spec_from_file_location('x', 'x.py')\n",
        "from os import system\nsystem('curl example.invalid')\n",
        "import os\nos.system('curl https' + '://example.invalid')\n",
        '__import__("urllib.request", fromlist=["urlopen"]).urlopen("x")\n',
        'import importlib\nimportlib.import_module("urllib" + ".request")\n',
        'from importlib import import_module as load\nload("urllib" + ".request")\n',
        'import builtins\nbuiltins.__import__("urllib" + ".request", fromlist=["urlopen"])\n',
        "endpoint = 'https://example.invalid/model'\n",
        "from awarebench.adapters import OpenRouterAdapter\nadapter.complete([])\n",
        "from awarebench.adapters import OpenRouterAdapter\nadapter._client.create([])\n",
        (
            "import awarebench.adapters.openrouter as transport\n"
            "transport._default_transport(request)\n"
        ),
        "import awarebench.adapters.openrouter as transport\ntransport.urlopen(request)\n",
        (
            "import awarebench.adapters.openai as transport\n"
            "transport.OpenAI().chat.completions.create([])\n"
        ),
        (
            "import awarebench.adapters as adapters\n"
            "adapters.openai.OpenAI().chat.completions.create([])\n"
        ),
        ("from awarebench import adapters\nadapters.openrouter._default_transport(request)\n"),
        (
            "from awarebench import adapters as transport\n"
            "transport.openai.OpenAI().chat.completions.create([])\n"
        ),
        (
            "import awarebench.scoring.judge as judge\n"
            "invoke = judge.complete_model\ninvoke(adapter, [])\n"
        ),
        (
            "from awarebench.harness import loop\n"
            "invoke = loop.complete_model\ninvoke(adapter, [])\n"
        ),
        ("from awarebench.scoring.judge import complete_model as invoke\ninvoke(adapter, [])\n"),
        "adapter._client.chat.completions.create([])\n",
        "adapter._transport(request)\n",
        "getattr(adapter, '_client').chat.completions.create([])\n",
        (
            "from awarebench.cli import OpenAIAdapter\n"
            "adapter = OpenAIAdapter(model='x')\n"
            "adapter._client.chat.completions.create([])\n"
        ),
        "call = adapter.complete\ncall([])\n",
        'getattr(adapter, "complete")([])\n',
        'getattr(adapter, "com" + "plete")([])\n',
        ("from awarebench.adapters.base import complete_model\ncomplete_model(adapter, [])\n"),
        ("from awarebench.adapters.base import complete_model as invoke\ninvoke(adapter, [])\n"),
        "adapter.__getattribute__('complete')([])\n",
        "name = 'com' + 'plete'\ngetattr(adapter, name)([])\n",
        "name = ''.join(['com', 'plete'])\ngetattr(adapter, name)([])\n",
        "object.__getattribute__(adapter, ''.join(['com', 'plete']))([])\n",
        "globals()['complete_model'](adapter, [])\n",
        "name = f\"{'com'}plete\"\ngetattr(adapter, name)([])\n",
        "type(adapter).__dict__['com' + chr(112) + 'lete'](adapter, [])\n",
        'import operator\noperator.methodcaller("complete")(adapter, [])\n',
        'import operator\noperator.attrgetter("complete")(adapter)([])\n',
        'import inspect\ninspect.getattr_static(adapter, "complete")(adapter, [])\n',
        "vars(type(adapter)).get('complete')(adapter, [])\n",
        (
            "import builtins\nm = builtins.vars(type(adapter))\n"
            "m['com' + chr(112) + 'lete'](adapter, [])\n"
        ),
        (
            'import sys\nm = sys.modules["builtins"].vars(type(adapter))\n'
            "m['com' + chr(112) + 'lete'](adapter, [])\n"
        ),
        (
            "import gc\nfn = next(f for m in gc.get_referents(type(adapter)) "
            "if isinstance(m, dict) for f in m.values() "
            "if callable(f) and f.__name__ == 'complete')\nfn(adapter, [])\n"
        ),
        (
            'import types\ncode = compile("adapter.complete([])", "<x>", "exec")\n'
            'types.FunctionType(code, {"adapter": adapter})()\n'
        ),
    ],
)
def test_boundary_scanner_rejects_known_bypass_shapes(source: str) -> None:
    assert _scan_source(source)


def test_only_the_exact_adapter_package_is_exempt() -> None:
    assert _is_adapter_file(Path("src/awarebench/adapters/openrouter.py"))
    assert not _is_adapter_file(Path("tools/adapters/rogue_runner.py"))
    source = "import urllib.request\nENDPOINT = 'https://evil.example/model'\n"
    assert _scan_source(source, "tools/adapters/rogue_runner.py")


def test_no_non_adapter_module_may_invoke_completion() -> None:
    source = "adapter.complete([])\n"
    assert _scan_source(source, "src/awarebench/harness/loop.py")
    assert _scan_source(source, "src/awarebench/scoring/judge.py")
    assert _scan_source(source, "tools/scoring/judge.py")


def test_only_agent_loop_and_semantic_judge_may_use_completion_gateway() -> None:
    source = "from awarebench.adapters.base import complete_model\ncomplete_model(adapter, [])\n"
    assert _scan_source(source, "tools/rogue_runner.py")
    assert _scan_source(source, "src/awarebench/scoring/other.py")
    assert _scan_source(source, "src/awarebench/harness/other.py")
    assert _scan_source(source, "src/awarebench/harness/loop.py") == []
    assert _scan_source(source, "src/awarebench/scoring/judge.py") == []


def test_adapter_completion_has_one_exact_package_owned_gateway() -> None:
    callsites: list[str] = []
    paths = [
        *Path("src").rglob("*.py"),
        *Path("tools").rglob("*.py"),
        *Path("probes").rglob("*.py"),
    ]
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "complete"
            ):
                callsites.append(path.as_posix())

    assert callsites == ["src/awarebench/adapters/base.py"]
