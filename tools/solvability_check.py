"""Compatibility wrapper for the package-owned solvability command."""

from __future__ import annotations

import sys

from awarebench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["solvability", *sys.argv[1:]]))
