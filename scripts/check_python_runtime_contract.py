#!/usr/bin/env python3
"""Enforce explicit, independent Python contracts for app and VLM service."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
violations: list[str] = []

# if sys.version_info < (3, 11):
#     violations.append(
#         f"Application quality checks require Python 3.11+, got {sys.version.split()[0]}"
#     )

app_config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
app_requires = app_config.get("project", {}).get("requires-python")
app_ruff = app_config.get("tool", {}).get("ruff", {}).get("target-version")
app_mypy = app_config.get("tool", {}).get("mypy", {}).get("python_version")

if app_requires != ">=3.11":
    violations.append(f"Application requires-python must be >=3.11, got {app_requires!r}")
if app_ruff != "py311":
    violations.append(f"Application Ruff target must be py311, got {app_ruff!r}")
if app_mypy != "3.11":
    violations.append(f"Application Mypy target must be 3.11, got {app_mypy!r}")

python_version_file = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
if python_version_file != "3.11":
    violations.append(f"Application .python-version must pin 3.11, got {python_version_file!r}")

vlm_config = tomllib.loads(
    (ROOT / "services" / "receipt-vlm" / "pyproject.toml").read_text(encoding="utf-8")
)
vlm_requires = vlm_config.get("project", {}).get("requires-python")
vlm_ruff = vlm_config.get("tool", {}).get("ruff", {}).get("target-version")
if vlm_requires != ">=3.10,<3.11":
    violations.append(
        "VLM service requires-python must preserve the known runtime range "
        f">=3.10,<3.11, got {vlm_requires!r}"
    )
if vlm_ruff != "py310":
    violations.append(f"VLM service Ruff target must be py310, got {vlm_ruff!r}")

app_runtime = (ROOT / "docker" / "Dockerfile.app-runtime").read_text(encoding="utf-8")
vlm_runtime = (ROOT / "docker" / "Dockerfile.vlm-runtime-cu126").read_text(encoding="utf-8")
if not re.search(r"^FROM python:3\.11(?:-|$)", app_runtime, flags=re.MULTILINE):
    violations.append("App runtime must derive from Python 3.11")
for required in ("python3", "python3-pip", "python3-dev"):
    if required not in vlm_runtime:
        violations.append(f"VLM runtime must retain Ubuntu Python 3.10 package {required}")
for forbidden in ("uv python install 3.11", "sys.version_info >= (3, 11)"):
    if forbidden in vlm_runtime:
        violations.append(f"VLM runtime must not inherit the app Python baseline: {forbidden}")

active_files = (
    "docker-compose.yml",
    "docker-compose.build.yml",
    "docker/Dockerfile.vlm-app",
    "docker/Dockerfile.vlm-gpu-cu126",
    "scripts/docker/build-vlm-runtime.ps1",
    "scripts/docker/build-vlm.ps1",
    ".github/workflows/docker-build.yml",
)
for relative in active_files:
    source = (ROOT / relative).read_text(encoding="utf-8")
    if "receipt-vlm-runtime:py311-cu126" in source:
        violations.append(f"{relative} still references the abandoned Python 3.11 VLM runtime")
    if "paddle-gemma-receipt-vlm:py311-gpu-cu126" in source:
        violations.append(f"{relative} still references the abandoned Python 3.11 VLM image")

if violations:
    print("Python runtime contract violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Runtime contracts passed: application=Python 3.11, receipt-vlm=Python 3.10.")
