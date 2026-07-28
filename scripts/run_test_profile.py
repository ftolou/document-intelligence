#!/usr/bin/env python3
"""Run the stable pytest profiles."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "unit": ["tests/unit"],
    "integration": ["tests/integration"],
    "regression": ["-m", "regression", "tests/regression"],
    "fast": ["-m", "not gpu and not ollama", "tests"],
    "all": ["tests"],
}


def main() -> int:
    profile = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if profile not in PROFILES:
        choices = ", ".join(sorted(PROFILES))
        print(f"Unknown test profile {profile!r}. Choose one of: {choices}")
        return 2
    command = [sys.executable, "-m", "pytest", *PROFILES[profile], *sys.argv[2:]]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
