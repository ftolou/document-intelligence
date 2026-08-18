#!/usr/bin/env python3
"""Run the repository test suite with pytest."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if importlib.util.find_spec("pytest") is None:
    raise SystemExit(
        "pytest is not installed. "
        "Install development dependencies with: "
        "python -m pip install -r requirements/app.txt -r requirements/dev.txt"
    )

command = [sys.executable, "-m", "pytest", *sys.argv[1:]]
raise SystemExit(subprocess.call(command, cwd=ROOT))
