"""Repository-level guards for the published source tree."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _tracked_file_paths() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )

    return tuple(path.decode("utf-8") for path in result.stdout.split(b"\0") if path)


def _tracked_dockerfile_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in paths if Path(path).name.lower().startswith("dockerfile"))


def _text_paths_with_raw_carriage_returns(
    paths: tuple[str, ...], read_bytes: Callable[[str], bytes]
) -> tuple[str, ...]:
    offenders: list[str] = []
    for path in paths:
        content = read_bytes(path)
        if b"\0" in content:
            continue
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if b"\r" in content:
            offenders.append(path)
    return tuple(offenders)


def test_tracked_files_contain_no_em_dashes() -> None:
    result = subprocess.run(
        ["git", "grep", "-n", "--", "\N{EM DASH}"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stdout


def test_tracked_files_contain_no_dockerfiles() -> None:
    assert _tracked_dockerfile_paths(_tracked_file_paths()) == ()


def test_tracked_text_files_contain_no_raw_carriage_returns() -> None:
    paths = _tracked_file_paths()
    offenders = _text_paths_with_raw_carriage_returns(
        paths,
        lambda path: (REPOSITORY_ROOT / path).read_bytes(),
    )

    assert offenders == ()


def test_dockerfile_guard_identifies_dockerfile_names() -> None:
    paths = ("probes/example/env/Dockerfile", "Dockerfile.dev", "README.md")

    assert _tracked_dockerfile_paths(paths) == (
        "probes/example/env/Dockerfile",
        "Dockerfile.dev",
    )


def test_text_byte_guard_identifies_raw_carriage_returns_safely() -> None:
    contents = {
        "clean.md": b"line one\nline two\n",
        "crlf.md": b"line one\r\n",
        "raw-cr.md": b"line one\rline two\n",
        "binary.dat": b"\x00\r\n",
        "non-utf8.txt": b"\xff\r\n",
    }

    assert _text_paths_with_raw_carriage_returns(tuple(contents), contents.__getitem__) == (
        "crlf.md",
        "raw-cr.md",
    )
