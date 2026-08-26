"""Fail-closed scan of repository and agent-visible leakage surfaces."""

from __future__ import annotations

import ast
import getpass
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TextIO

import yaml

from awarebench.cli import _build_stack
from awarebench.events import EventLog
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.probes.loader import ProbeGateError, load_probe

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
VARIANTS: Final[tuple[str, ...]] = ("fault", "control")
HANDLER_OUTPUT_SAMPLES: Final[int] = 3
GENERIC_IDENTITIES: Final[frozenset[str]] = frozenset(
    {
        "admin",
        "administrator",
        "agent",
        "build",
        "ci",
        "github",
        "gitlab",
        "root",
        "runner",
        "service",
        "user",
        "worktrees",
    }
)
IGNORED_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {".git", ".local", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
)
LEAK_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("benchmark", re.compile(r"\bbenchmark\b", re.IGNORECASE)),
    ("probe", re.compile(r"\bprobe\b", re.IGNORECASE)),
    ("injected", re.compile(r"\binjected\b", re.IGNORECASE)),
    ("ground_truth", re.compile(r"\bground[_\s]?truth\b", re.IGNORECASE)),
    ("awarebench", re.compile(r"\bawarebench\b", re.IGNORECASE)),
    ("awareness_scored", re.compile(r"\bawareness[_\s]score\b", re.IGNORECASE)),
)
LOCAL_PATH_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("local_path", re.compile(re.escape("/" + "home" + "/"), re.IGNORECASE)),
    ("local_path", re.compile(re.escape("/" + "Users" + "/"), re.IGNORECASE)),
    ("local_path", re.compile(re.escape("C" + ":" + "\\"), re.IGNORECASE)),
    ("local_path", re.compile(re.escape("~" + "/"))),
)


class ScanError(RuntimeError):
    """The scan could not inspect a required surface."""


@dataclass
class ScanResult:
    """All scanner observations, including no-op-proof coverage counts."""

    files_inspected: int = 0
    probes_inspected: int = 0
    environments_inspected: int = 0
    runtime_files_inspected: int = 0
    command_outputs_inspected: int = 0
    leaks: list[str] = field(default_factory=list)


def _identity_patterns(root: Path) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Return local-path-qualified patterns for non-generic machine identities."""
    username = getpass.getuser().casefold()
    parent = root.parent.name.casefold()
    candidates = tuple(
        candidate
        for candidate in (username, parent)
        if candidate and candidate not in GENERIC_IDENTITIES
    )
    posix_home = re.escape("/" + "home" + "/")
    posix_users = re.escape("/" + "Users" + "/")
    windows_users = r"[A-Za-z]:" + re.escape("\\" + "Users" + "\\")
    user_prefix = f"(?:{posix_home}|{posix_users}|{windows_users})"
    parent_prefix = "(?:" + posix_home + "[^/]+/|" + posix_users + "[^/]+/|" + windows_users
    parent_prefix += r"[^\\]+\\)"
    patterns: list[tuple[str, re.Pattern[str]]] = []
    if username in candidates:
        patterns.append(
            (
                "local_identity",
                re.compile(user_prefix + re.escape(username) + r"(?=[/\\]|$)", re.IGNORECASE),
            )
        )
    if parent in candidates:
        patterns.append(
            (
                "local_identity",
                re.compile(parent_prefix + re.escape(parent) + r"(?=[/\\]|$)", re.IGNORECASE),
            )
        )
    return tuple(patterns)


def _agent_visible_patterns(root: Path) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Combine evaluation, local path, and path-qualified identity checks."""
    return LEAK_PATTERNS + LOCAL_PATH_PATTERNS + _identity_patterns(root)


def _check_text(
    text: str,
    source: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> list[str]:
    return [
        f"{source}: pattern '{label}' matched"
        for label, pattern in patterns
        if pattern.search(text)
    ]


def _repository_files(root: Path) -> list[Path]:
    """Return every regular repository file except generated local state."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in IGNORED_DIRECTORIES for part in path.relative_to(root).parts)
    )


def _scan_local_paths(root: Path, result: ScanResult) -> None:
    """Scan all repository text, not only probe artifacts, for machine identity."""
    patterns = LOCAL_PATH_PATTERNS + _identity_patterns(root)
    for path in _repository_files(root):
        result.files_inspected += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise ScanError(f"cannot read repository file {path}: {exc}") from exc
        result.leaks.extend(_check_text(text, str(path), patterns))


def _task_text(manifest_path: Path) -> str:
    """Read task text and make malformed manifests terminal scanner errors."""
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ScanError(f"cannot parse manifest {manifest_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ScanError(f"cannot parse manifest {manifest_path}: expected a mapping")
    task = data.get("task")
    return task if isinstance(task, str) else ""


def _python_string_literals(path: Path) -> list[str]:
    """Read every Python string literal, preserving the previous source scan surface."""
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ScanError(f"cannot parse Python source {path}: {exc}") from exc
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(module)
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Module))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _scan_source_literals(probe_dir: Path, result: ScanResult) -> None:
    """Scan probe artifact literals that can feed agent-visible runtime content."""
    try:
        loaded = load_probe(probe_dir)
    except ProbeGateError as exc:
        raise ScanError(f"cannot load probe source {probe_dir}: {exc}") from exc
    for path in (loaded.generator, loaded.injection, loaded.control):
        for literal in _python_string_literals(path):
            result.leaks.extend(_check_text(literal, f"{path} [source literal]", LEAK_PATTERNS))


def _scan_environment(
    root: Path,
    probe_dir: Path,
    seed: int,
    variant: str,
    result: ScanResult,
) -> None:
    """Construct one agent-visible environment and scan its files and commands."""
    try:
        loaded = load_probe(probe_dir)
        log = EventLog()
        clock = VirtualClock()
        cycles = CycleCounter()
        parts = _build_stack(loaded, log, clock, cycles, seed=seed, variant=variant)
    except (ImportError, OSError, ProbeGateError, RuntimeError, ValueError) as exc:
        raise ScanError(f"cannot instantiate {probe_dir} ({variant}, seed {seed}): {exc}") from exc

    result.environments_inspected += 1
    source = f"{probe_dir} [runtime {variant} seed={seed}]"
    patterns = _agent_visible_patterns(root)
    for virtual_path in parts.fs.list_files():
        content = parts.fs.read(virtual_path)
        if content is None:
            raise ScanError(f"runtime file disappeared in {probe_dir}: {virtual_path}")
        result.runtime_files_inspected += 1
        result.leaks.extend(_check_text(content, f"{source} {virtual_path}", patterns))

    for command, handler in sorted(parts.command_handlers.items()):
        for output_index in range(HANDLER_OUTPUT_SAMPLES):
            try:
                _exit_code, output = handler(command)
            except Exception as exc:
                raise ScanError(
                    f"cannot inspect command output for {probe_dir} "
                    f"({variant}, seed {seed}, {command}, output {output_index + 1}): {exc}"
                ) from exc
            result.command_outputs_inspected += 1
            result.leaks.extend(_check_text(output, f"{source} command={command}", patterns))


def _scan_probe(root: Path, probe_dir: Path, result: ScanResult) -> None:
    manifest_path = probe_dir / "probe.yaml"
    result.probes_inspected += 1
    result.leaks.extend(
        _check_text(_task_text(manifest_path), f"{manifest_path} [task]", LEAK_PATTERNS)
    )
    _scan_source_literals(probe_dir, result)
    for seed in SEEDS:
        for variant in VARIANTS:
            _scan_environment(root, probe_dir, seed, variant, result)


def scan_root(root: Path) -> ScanResult:
    """Scan all local paths and every discovered probe's visible surfaces."""
    root = root.resolve()
    if not root.is_dir():
        raise ScanError(f"scan root is not a directory: {root}")
    result = ScanResult()
    _scan_local_paths(root, result)
    for manifest_path in sorted(root.rglob("probe.yaml")):
        if any(part in IGNORED_DIRECTORIES for part in manifest_path.relative_to(root).parts):
            continue
        _scan_probe(root, manifest_path.parent, result)
    return result


def _report_counts(result: ScanResult, *, stream: TextIO) -> None:
    print(
        "inspected "
        f"{result.files_inspected} files, {result.probes_inspected} probes, "
        f"{result.environments_inspected} environments, "
        f"{result.runtime_files_inspected} virtual files, and "
        f"{result.command_outputs_inspected} command outputs.",
        file=stream,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the scanner, returning 0 clean, 1 leaks, or 2 inspection failure."""
    args = argv if argv is not None else sys.argv[1:]
    root = Path(args[0]) if args else Path.cwd()
    try:
        result = scan_root(root)
    except ScanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if result.files_inspected == 0 or result.probes_inspected == 0:
        _report_counts(result, stream=sys.stderr)
        print("ERROR: scan inspected zero required files or probes.", file=sys.stderr)
        return 2
    if result.leaks:
        for leak in result.leaks:
            print(f"LEAK: {leak}", file=sys.stderr)
        _report_counts(result, stream=sys.stderr)
        print(f"{len(result.leaks)} leakage violation(s) found.", file=sys.stderr)
        return 1

    _report_counts(result, stream=sys.stdout)
    print("Leakage scan clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
