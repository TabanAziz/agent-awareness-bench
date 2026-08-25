"""Scans agent-visible surfaces for evaluation-revealing vocabulary.

Scans probe.yaml task text, handler stdout patterns in generator/injection/
control source code, and virtual file templates for strings that could reveal
the benchmark to a tested agent.

Exit code 0 = clean; 1 = leaks found (each printed to stderr).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Vocabulary that would reveal the eval if present in agent-visible output.
LEAK_PATTERNS: list[tuple[str, str]] = [
    ("benchmark", r"\bbenchmark\b"),
    ("probe", r"\bprobe\b"),
    ("injected", r"\binjected\b"),
    ("ground_truth", r"\bground[_\s]?truth\b"),
    ("awareness_bench", r"\bawarebench\b"),
    ("awareness_scored", r"\bawareness[_\s]score\b"),
]

# Files that produce agent-visible content.
AGENT_VISIBLE_GLOBS: list[str] = [
    "**/generator.py",
    "**/injection.py",
    "**/control.py",
]


def _check_text(text: str, source: str) -> list[str]:
    """Return leak descriptions found in text."""
    lowered = text.lower()
    return [
        f"{source}: pattern '{label}' matched"
        for label, pattern in LEAK_PATTERNS
        if re.search(pattern, lowered)
    ]


def scan_probe(probe_dir: Path) -> list[str]:
    """Scan all agent-visible surfaces of one probe directory."""
    leaks: list[str] = []

    manifest_path = probe_dir / "probe.yaml"
    if manifest_path.exists():
        leaks.extend(_check_text(manifest_path.read_text(encoding="utf-8"), str(manifest_path)))

    for pattern in AGENT_VISIBLE_GLOBS:
        for py_file in sorted(probe_dir.glob(pattern)):
            if py_file.exists():
                content = py_file.read_text(encoding="utf-8")
                # Strip docstrings and comments: they are maintainer-facing.
                stripped = re.sub(r'""".*?"""', "", content, flags=re.DOTALL)
                stripped = re.sub(r"#[^\n]*", "", stripped)
                # Check string literals only.
                strings = re.findall(r'"([^"]*)"', stripped)
                joined = " ".join(strings).lower()
                leaks.extend(_check_text(joined, str(py_file)))

    return leaks


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is not None else sys.argv[1:]
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
