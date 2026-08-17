#!/usr/bin/env python3
"""Run repository quality checks for the canonical application runtime."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "src/receipt_intelligence",
    "scripts/run_receipt_pipeline.py",
    "scripts/run_receipt_images_folder.py",
    "tests",
    "scripts/run_tests.py",
    "scripts/run_test_profile.py",
    "scripts/check_dependency_compatibility.py",
    "scripts/check_python_runtime_contract.py",
    "scripts/check_security_boundaries.py",
    "scripts/check_configuration_contracts.py",
    "scripts/check_model_boundaries.py",
    "scripts/check_extraction_state_boundaries.py",
    "scripts/check_persistence_boundaries.py",
    "scripts/check_application_boundaries.py",
    "scripts/check_job_dispatch_boundaries.py",
    "scripts/check_rag_sql_composition.py",
    "scripts/check_observability_boundaries.py",
    "scripts/check_model_call_observability.py",
    "scripts/run_quality_checks.py",
    "scripts/verify_runtime_layout.py",
]

if importlib.util.find_spec("ruff") is None:
    raise SystemExit("Ruff is not installed. Run: python -m pip install -r requirements/dev.txt")

commands = [
    [sys.executable, "scripts/check_python_runtime_contract.py"],
    [sys.executable, "scripts/check_security_boundaries.py"],
    [sys.executable, "scripts/check_configuration_contracts.py"],
    [sys.executable, "scripts/check_model_boundaries.py"],
    [sys.executable, "scripts/check_extraction_state_boundaries.py"],
    [sys.executable, "scripts/check_persistence_boundaries.py"],
    [sys.executable, "scripts/check_application_boundaries.py"],
    [sys.executable, "scripts/check_job_dispatch_boundaries.py"],
    [sys.executable, "scripts/check_rag_sql_composition.py"],
    [sys.executable, "scripts/check_observability_boundaries.py"],
    [sys.executable, "scripts/check_model_call_observability.py"],
    [sys.executable, "-m", "ruff", "check", *TARGETS],
    [sys.executable, "-m", "ruff", "format", "--check", *TARGETS],
]

for command in commands:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

print("Quality checks passed.")
