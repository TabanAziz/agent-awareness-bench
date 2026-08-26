"""Scans agent-visible surfaces for evaluation-revealing vocabulary.

Scans ONLY the task text from probe.yaml (the sole string that reaches the
agent via the system prompt) and string literals in generator/injection/
control source code that produce handler stdout.

Exit code 0 = clean; 1 = leaks found (each printed to stderr).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

LEAK_PATTERNS: list[tuple[str, str]] = [
    ("benchmark", r"\bbenchmark\b"),
    ("probe", r"\bprobe\b"),
    ("injected", r"\binjected\b"),
    ("ground_truth", r"\bground[_\s]?truth\b"),
    ("awarebench", r"\bawarebench\b"),
    ("awareness_scored", r"\bawareness[_\s]score\b"),
]


def _check_text(text: str, source: str) -> list[str]:
    lowered = text.lower()
    return [
        f"{source}: pattern '{label}' matched"
        for label, pattern in LEAK_PATTERNS
        if re.search(pattern, lowered)
    ]


def _extract_task_text(manifest_path: Path) -> str:
    """Extract only the agent-visible task field from probe.yaml."""
    if yaml is None:
        return ""
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return str(data.get("task", ""))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read manifest: {manifest_path}") from exc
    return ""


def _extract_handler_strings(py_path: Path) -> str:
    """Extract string literals from injection.py and control.py that become stdout."""
    if not py_path.exists():
        return ""
    content = py_path.read_text(encoding="utf-8")
    # Extract JSON dump values and f-string content that becomes stdout
    strings = re.findall(r'"([^"]*)"', content)
    return " ".join(strings).lower()


def scan_probe(probe_dir: Path) -> list[str]:
    """Scan all agent-visible surfaces of one probe directory."""
    leaks: list[str] = []

    # Scan task text (the primary agent-visible surface)
    manifest_path = probe_dir / "probe.yaml"
    if manifest_path.exists():
        task_text = _extract_task_text(manifest_path)
        if task_text:
            leaks.extend(_check_text(task_text.lower(), f"{manifest_path} [task]"))

    # Scan handler stdout templates from injection.py (the fault arm)
    for role in ("injection",):
        py_path = probe_dir / f"{role}.py"
        if py_path.exists():
            handler_strings = _extract_handler_strings(py_path)
            leaks.extend(_check_text(handler_strings, f"{py_path} [stdout]"))

    return leaks


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    probes_root = Path(args[0]) if args else Path("probes")

    all_leaks: list[str] = []
    for probe_yaml in sorted(probes_root.rglob("probe.yaml")):
        all_leaks.extend(scan_probe(probe_yaml.parent))

    if all_leaks:
        for leak in all_leaks:
            print(f"LEAK: {leak}", file=sys.stderr)
        print(f"\n{len(all_leaks)} leakage violation(s) found.", file=sys.stderr)
        return 1

    print("Leakage scan clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
