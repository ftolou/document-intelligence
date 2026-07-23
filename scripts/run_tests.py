#!/usr/bin/env python3
"""Run the test suite with pytest when available, otherwise unittest."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if importlib.util.find_spec("pytest") is not None:
    command = [sys.executable, "-m", "pytest", *sys.argv[1:]]
else:
    if len(sys.argv) > 1:
        raise SystemExit("Extra pytest arguments require pytest from requirements/dev.txt.")
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
        "-v",
    ]

raise SystemExit(subprocess.call(command, cwd=ROOT))
