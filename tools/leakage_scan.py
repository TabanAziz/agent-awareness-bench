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
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.loop import DEFAULT_CYCLE_STEP_US
from awarebench.harness.stack import StackParts
from awarebench.harness.tools import ToolHost
from awarebench.probes.loader import ProbeGateError, load_probe

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
VARIANTS: Final[tuple[str, ...]] = ("fault", "control")
HANDLER_OUTPUT_SAMPLES: Final[int] = 40
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
    stack_instantiations: int = 0
    schedules_inspected: int = 0
    command_samples_declared: int = 0
    seed_messages_inspected: int = 0
    http_bodies_inspected: int = 0
    virtual_filenames_inspected: int = 0
    runtime_files_inspected: int = 0
    virtual_state_rescans: int = 0
    command_outputs_inspected: int = 0
    binary_files_skipped: int = 0
    leaks: list[str] = field(default_factory=list)


def _identity_patterns(root: Path) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Return bare-word patterns for non-generic local machine identities."""
    username = getpass.getuser().casefold()
    parent = root.parent.name.casefold()
    repository_name = root.name.casefold()
    candidates = tuple(
        candidate
        for candidate in (username, parent if parent != repository_name else "")
        if candidate and candidate not in GENERIC_IDENTITIES
    )
    return tuple(
        (
            "local_identity",
            re.compile(
                r"(?<![A-Za-z0-9_.-])" + re.escape(candidate) + r"(?![A-Za-z0-9_.-])", re.IGNORECASE
            ),
        )
        for candidate in candidates
    )


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
        try:
            text = _decode_repository_text(path)
        except OSError as exc:
            raise ScanError(f"cannot read repository file {path}: {exc}") from exc
        if text is None:
            result.binary_files_skipped += 1
            continue
        result.files_inspected += 1
        result.leaks.extend(_check_text(text, str(path), patterns))


def _decode_repository_text(path: Path) -> str | None:
    """Decode UTF-8 and BOM or NUL-patterned UTF-16; skip opaque binary data."""
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if _looks_like_utf16_without_bom(data):
        encoding = "utf-16-le" if data[1::2].count(0) >= data[::2].count(0) else "utf-16-be"
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _looks_like_utf16_without_bom(data: bytes) -> bool:
    """Identify interleaved-NUL UTF-16 before permissive UTF-8 decoding."""
    if len(data) < 4 or len(data) % 2:
        return False
    even = data[::2]
    odd = data[1::2]
    return even.count(0) * 2 >= len(even) or odd.count(0) * 2 >= len(odd)


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
    """Scan one seed/variant through shared and per-handler fresh schedules."""
    result.environments_inspected += 1
    parts, log, clock, cycles = _instantiate_stack(probe_dir, seed, variant, result)
    samples_by_handler = _command_samples(parts, probe_dir, variant, seed)
    result.command_samples_declared += sum(len(samples) for samples in samples_by_handler.values())
    handlers = sorted(samples_by_handler)
    round_robin_samples = tuple(
        sample for handler in handlers for sample in samples_by_handler[handler]
    )
    if parts.http_table:
        # HTTP has a distinct agent-reachable schedule.  Keep every command
        # schedule at cycles 1..40, matching AgentLoop's run horizon exactly.
        _scan_schedule(
            root,
            probe_dir,
            seed,
            variant,
            parts,
            log,
            clock,
            cycles,
            "http",
            (),
            sample_http=True,
            result=result,
        )

    if round_robin_samples:
        if parts.http_table:
            parts, log, clock, cycles = _instantiate_stack(probe_dir, seed, variant, result)
            _require_sample_corpus_matches(samples_by_handler, parts, probe_dir, variant, seed)
        _scan_schedule(
            root,
            probe_dir,
            seed,
            variant,
            parts,
            log,
            clock,
            cycles,
            "round-robin",
            tuple(
                round_robin_samples[index % len(round_robin_samples)]
                for index in range(HANDLER_OUTPUT_SAMPLES)
            ),
            sample_http=False,
            result=result,
        )
    elif not parts.http_table:
        _scan_schedule(
            root,
            probe_dir,
            seed,
            variant,
            parts,
            log,
            clock,
            cycles,
            "surfaces-only",
            (),
            sample_http=False,
            result=result,
        )

    for command in handlers:
        for sample in samples_by_handler[command]:
            fresh_parts, fresh_log, fresh_clock, fresh_cycles = _instantiate_stack(
                probe_dir, seed, variant, result
            )
            _require_handlers(fresh_parts, (command,), probe_dir, variant, seed)
            _require_sample_corpus_matches(
                samples_by_handler, fresh_parts, probe_dir, variant, seed
            )
            _scan_schedule(
                root,
                probe_dir,
                seed,
                variant,
                fresh_parts,
                fresh_log,
                fresh_clock,
                fresh_cycles,
                f"handler={command} sample={sample}",
                (sample,) * HANDLER_OUTPUT_SAMPLES,
                sample_http=False,
                result=result,
            )

    for mutator in handlers:
        for reader in handlers:
            if mutator == reader:
                continue
            for mutator_sample in samples_by_handler[mutator]:
                for reader_sample in samples_by_handler[reader]:
                    for cutover in range(1, HANDLER_OUTPUT_SAMPLES):
                        fresh_parts, fresh_log, fresh_clock, fresh_cycles = _instantiate_stack(
                            probe_dir, seed, variant, result
                        )
                        _require_handlers(fresh_parts, (mutator, reader), probe_dir, variant, seed)
                        _require_sample_corpus_matches(
                            samples_by_handler, fresh_parts, probe_dir, variant, seed
                        )
                        _scan_schedule(
                            root,
                            probe_dir,
                            seed,
                            variant,
                            fresh_parts,
                            fresh_log,
                            fresh_clock,
                            fresh_cycles,
                            (
                                f"cutover={mutator}({mutator_sample})-to-"
                                f"{reader}({reader_sample}) k={cutover}"
                            ),
                            (mutator_sample,) * cutover + (reader_sample,),
                            sample_http=False,
                            result=result,
                        )


def _require_handlers(
    parts: StackParts,
    required: tuple[str, ...],
    probe_dir: Path,
    variant: str,
    seed: int,
) -> None:
    """Fail closed if a fresh schedule lacks a handler discovered initially."""
    for command in required:
        if command not in parts.command_handlers:
            raise ScanError(
                f"cannot inspect {probe_dir} ({variant}, seed {seed}): "
                f"handler {command!r} is absent from its fresh schedule"
            )


def _command_samples(
    parts: StackParts, probe_dir: Path, variant: str, seed: int
) -> dict[str, tuple[str, ...]]:
    """Validate the explicit full-command corpus against ToolHost prefix rules."""
    handlers = set(parts.command_handlers)
    corpus_keys = set(parts.command_samples)
    missing = sorted(handlers - corpus_keys)
    extra = sorted(corpus_keys - handlers)
    if missing or extra:
        raise ScanError(
            f"invalid command samples for {probe_dir} ({variant}, seed {seed}): "
            f"missing={missing}, extra={extra}"
        )
    validated: dict[str, tuple[str, ...]] = {}
    for prefix in sorted(handlers):
        samples = parts.command_samples[prefix]
        if not samples or not all(isinstance(sample, str) and sample for sample in samples):
            raise ScanError(
                f"invalid command samples for {probe_dir} ({variant}, seed {seed}): "
                f"handler {prefix!r} requires one or more non-empty strings"
            )
        for sample in samples:
            matches = [candidate for candidate in handlers if sample.startswith(candidate)]
            resolved = max(matches, key=len) if matches else None
            if resolved != prefix:
                raise ScanError(
                    f"invalid command samples for {probe_dir} ({variant}, seed {seed}): "
                    f"sample {sample!r} does not resolve to handler {prefix!r}"
                )
        validated[prefix] = tuple(samples)
    return validated


def _require_sample_corpus_matches(
    expected: dict[str, tuple[str, ...]],
    parts: StackParts,
    probe_dir: Path,
    variant: str,
    seed: int,
) -> None:
    """Fresh schedules must retain the exact validated command corpus."""
    actual = _command_samples(parts, probe_dir, variant, seed)
    if actual != expected:
        raise ScanError(
            f"command samples changed across fresh schedules for {probe_dir} "
            f"({variant}, seed {seed})"
        )


def _instantiate_stack(
    probe_dir: Path, seed: int, variant: str, result: ScanResult
) -> tuple[StackParts, EventLog, VirtualClock, CycleCounter]:
    """Build one fresh runtime stack and record the actual instantiation."""
    try:
        loaded = load_probe(probe_dir)
        log = EventLog()
        clock = VirtualClock()
        cycles = CycleCounter()
        parts = _build_stack(loaded, log, clock, cycles, seed=seed, variant=variant)
    except (ImportError, OSError, ProbeGateError, RuntimeError, ValueError) as exc:
        raise ScanError(f"cannot instantiate {probe_dir} ({variant}, seed {seed}): {exc}") from exc

    result.stack_instantiations += 1
    return parts, log, clock, cycles


def _scan_schedule(
    root: Path,
    probe_dir: Path,
    seed: int,
    variant: str,
    parts: StackParts,
    log: EventLog,
    clock: VirtualClock,
    cycles: CycleCounter,
    schedule: str,
    commands: tuple[str, ...],
    *,
    sample_http: bool,
    result: ScanResult,
) -> None:
    """Scan one fresh stack through exact ToolHost-visible values."""
    result.schedules_inspected += 1
    source = f"{probe_dir} [runtime {variant} seed={seed} schedule={schedule}]"
    patterns = _agent_visible_patterns(root)
    surfaces_inspected = 0
    host = ToolHost(
        event_log=log,
        clock=clock,
        cycles=cycles,
        budget=BudgetAccountant(),
        fs=parts.fs,
        faults=parts.faults,
        command_handlers=parts.command_handlers,
        http_table=parts.http_table,
        host_name="leakage-scan",
    )

    for message_index, (role, content) in enumerate(parts.seed_messages, start=1):
        result.seed_messages_inspected += 1
        surfaces_inspected += 1
        result.leaks.extend(
            _check_text(
                repr({"role": role, "content": content}),
                f"{source} seed-message={message_index}",
                patterns,
            )
        )

    def scan_virtual_files(state: str) -> None:
        nonlocal surfaces_inspected
        for virtual_path in parts.fs.list_files():
            content = parts.fs.read(virtual_path)
            if content is None:
                raise ScanError(f"runtime file disappeared in {probe_dir}: {virtual_path}")
            result.virtual_filenames_inspected += 1
            result.runtime_files_inspected += 1
            surfaces_inspected += 2
            result.leaks.extend(_check_text(virtual_path, f"{source} {state} filename", patterns))
            result.leaks.extend(_check_text(content, f"{source} {state} {virtual_path}", patterns))

    scan_virtual_files("initial")

    def advance() -> None:
        cycles.advance()
        clock.advance_us(DEFAULT_CYCLE_STEP_US)

    if sample_http:
        for url in sorted(parts.http_table):
            advance()
            try:
                http_response: tuple[str, int] = host.http_get(url)
            except Exception as exc:
                raise ScanError(
                    f"cannot inspect HTTP output for {probe_dir} "
                    f"({variant}, seed {seed}, url {url}): {exc}"
                ) from exc
            result.http_bodies_inspected += 1
            surfaces_inspected += 1
            result.leaks.extend(_check_text(repr(http_response), f"{source} http={url}", patterns))
            result.virtual_state_rescans += 1
            scan_virtual_files(f"after-http={url}")

    for step, command in enumerate(commands, start=1):
        advance()
        try:
            command_response: tuple[int, str] = host.run_command(command)
        except Exception as exc:
            raise ScanError(
                f"cannot inspect command output for {probe_dir} "
                f"({variant}, seed {seed}, step {step}, {command}): {exc}"
            ) from exc
        result.command_outputs_inspected += 1
        surfaces_inspected += 1
        result.leaks.extend(
            _check_text(repr(command_response), f"{source} command={command}", patterns)
        )
        result.virtual_state_rescans += 1
        scan_virtual_files(f"after-step={step} command={command}")

    if surfaces_inspected == 0:
        raise ScanError(
            f"{probe_dir} ({variant}, seed {seed}, schedule {schedule}) "
            "exposed zero agent-visible surfaces"
        )


def _scan_probe(root: Path, probe_dir: Path, result: ScanResult) -> None:
    manifest_path = probe_dir / "probe.yaml"
    result.probes_inspected += 1
    result.leaks.extend(
        _check_text(
            _task_text(manifest_path), f"{manifest_path} [task]", _agent_visible_patterns(root)
        )
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
        f"{result.stack_instantiations} stack instantiations, "
        f"{result.schedules_inspected} schedules, "
        f"{result.command_samples_declared} declared command samples, "
        f"{result.seed_messages_inspected} seed messages, "
        f"{result.http_bodies_inspected} HTTP bodies, "
        f"{result.virtual_filenames_inspected} virtual filenames, "
        f"{result.runtime_files_inspected} virtual file contents, "
        f"{result.virtual_state_rescans} virtual state rescans, "
        f"{result.command_outputs_inspected} command outputs, and "
        f"{result.binary_files_skipped} skipped binary files.",
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
