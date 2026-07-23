#!/usr/bin/env python3
"""Run scoped quality checks for the refactored application modules."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "src/receipt_intelligence/query",
    "src/receipt_intelligence/observability",
    "src/receipt_intelligence/web",
    "src/receipt_intelligence/services/artifact_service.py",
    "src/receipt_intelligence/services/job_processing.py",
    "src/receipt_intelligence/services/review_service.py",
    "src/receipt_intelligence/storage",
    "src/receipt_intelligence/runtime",
    "src/receipt_intelligence/extraction/__init__.py",
    "src/receipt_intelligence/extraction/artifacts.py",
    "src/receipt_intelligence/extraction/config.py",
    "src/receipt_intelligence/extraction/context.py",
    "src/receipt_intelligence/extraction/factory.py",
    "src/receipt_intelligence/extraction/stages",
    "src/receipt_intelligence/extraction/support.py",
    "src/receipt_intelligence/extraction/workflow.py",
    "src/receipt_intelligence/extraction/evidence/__init__.py",
    "src/receipt_intelligence/extraction/parsing/__init__.py",
    "src/receipt_intelligence/extraction/validation/__init__.py",
    "src/receipt_intelligence/extraction/repair/__init__.py",
    "src/receipt_intelligence/extraction/categorization/__init__.py",
    "src/receipt_intelligence/pipeline/integrated_receipt_pipeline.py",
    "scripts/run_receipt_folder.py",
    "scripts/run_receipt_images_folder.py",
    "tests",
    "scripts/run_tests.py",
    "scripts/run_test_profile.py",
    "scripts/check_dependency_compatibility.py",
    "scripts/run_quality_checks.py",
    "scripts/phase6_cutover.py",
    "scripts/verify_runtime_layout.py",
]

if importlib.util.find_spec("ruff") is None:
    raise SystemExit("Ruff is not installed. Run: python -m pip install -r requirements/dev.txt")

commands = [
    [sys.executable, "-m", "ruff", "check", *TARGETS],
    [sys.executable, "-m", "ruff", "format", "--check", *TARGETS],
]

for command in commands:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

print("RAG-SQL LangGraph quality checks passed.")
