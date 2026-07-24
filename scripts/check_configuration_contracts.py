#!/usr/bin/env python3
"""Fail CI when extraction configuration becomes permissive or ambiguous."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src" / "receipt_intelligence" / "extraction" / "config.py"
PIPELINE_PATH = (
    ROOT
    / "src"
    / "receipt_intelligence"
    / "pipeline"
    / "integrated_receipt_pipeline.py"
)
JOB_SERVICE_PATH = (
    ROOT / "src" / "receipt_intelligence" / "services" / "job_processing.py"
)

violations: list[str] = []

config_source = CONFIG_PATH.read_text(encoding="utf-8")
config_tree = ast.parse(config_source, filename=str(CONFIG_PATH))

for node in config_tree.body:
    if not isinstance(node, ast.ClassDef) or node.name != "ExtractionConfig":
        continue
    dataclass_call = next(
        (
            decorator
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "dataclass"
        ),
        None,
    )
    if dataclass_call is None:
        violations.append("ExtractionConfig must remain a dataclass")
        break
    keyword_values = {
        keyword.arg: keyword.value
        for keyword in dataclass_call.keywords
        if keyword.arg is not None
    }
    frozen = keyword_values.get("frozen")
    if not isinstance(frozen, ast.Constant) or frozen.value is not True:
        violations.append("ExtractionConfig must use dataclass(frozen=True)")
    break
else:
    violations.append("ExtractionConfig class is missing")

if "unused_kwargs" in config_source:
    violations.append("ExtractionConfig must not contain an unused_kwargs escape hatch")

pipeline_source = PIPELINE_PATH.read_text(encoding="utf-8")
if "**unused_kwargs" in pipeline_source:
    violations.append("The pipeline must not silently accept **unused_kwargs")
if "extraction_request_from_mapping" not in pipeline_source:
    violations.append("The compatibility entry point must use the strict argument mapper")

job_service_source = JOB_SERVICE_PATH.read_text(encoding="utf-8")
if "run_integrated_receipt_pipeline" in job_service_source:
    violations.append("Application job processing must use the typed extraction request API")
if (
    "ExtractionRequest(" not in job_service_source
    or "run_receipt_extraction(" not in job_service_source
):
    violations.append("Application job processing must construct and execute ExtractionRequest")

if violations:
    print("Configuration contract violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Extraction configuration contract checks passed.")
