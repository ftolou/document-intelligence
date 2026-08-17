#!/usr/bin/env python3
"""Enforce the single application Python runtime contract."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
violations: list[str] = []
config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

if config.get("project", {}).get("requires-python") != ">=3.11":
    violations.append("Application requires-python must be >=3.11")
if config.get("tool", {}).get("ruff", {}).get("target-version") != "py311":
    violations.append("Application Ruff target must be py311")
if config.get("tool", {}).get("mypy", {}).get("python_version") != "3.11":
    violations.append("Application Mypy target must be 3.11")
if (ROOT / ".python-version").read_text(encoding="utf-8").strip() != "3.11":
    violations.append(".python-version must pin 3.11")

runtime = (ROOT / "docker" / "Dockerfile.app-runtime").read_text(encoding="utf-8")
if not re.search(r"^FROM python:3\.11(?:-|$)", runtime, flags=re.MULTILINE):
    violations.append("App runtime must derive from Python 3.11")

for removed in (
    ROOT / "services" / "receipt-vlm" / "pyproject.toml",
    ROOT / "docker" / "Dockerfile.vlm-runtime-cu126",
):
    if removed.exists():
        violations.append(f"Removed runtime contract still exists: {removed.relative_to(ROOT)}")

if violations:
    print("Python runtime contract violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Runtime contract passed: application=Python 3.11.")
