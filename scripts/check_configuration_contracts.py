#!/usr/bin/env python3
"""Fail CI when extraction configuration becomes permissive or ambiguous."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src" / "receipt_intelligence" / "extraction" / "config.py"
PIPELINE_PATH = (
    ROOT / "src" / "receipt_intelligence" / "pipeline" / "integrated_receipt_pipeline.py"
)
JOB_SERVICE_PATH = ROOT / "src" / "receipt_intelligence" / "services" / "job_processing.py"
SETTINGS_PATH = ROOT / "src" / "receipt_intelligence" / "settings.py"

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
        keyword.arg: keyword.value for keyword in dataclass_call.keywords if keyword.arg is not None
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


env_example_source = (ROOT / ".env.example").read_text(encoding="utf-8")
for obsolete_name in (
    "EXTRACTION_STRATEGY",
    "SPATIAL_OVERVIEW_NUM_CTX",
    "SPATIAL_OVERVIEW_NUM_PREDICT",
    "SPATIAL_OVERVIEW_TIMEOUT_SECONDS",
    "MAX_REOCR_IMAGES",
    "VLM_ENABLED",
    "READINESS_PROBE_VLM",
    "READINESS_REQUIRE_VLM",
):
    if f"{obsolete_name}=" in env_example_source:
        violations.append(f".env.example contains obsolete setting: {obsolete_name}")

compose_source = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
if compose_source.count("env_file:") < 2:
    violations.append("Both Docker Compose services must load .env through env_file")
if "VLM_ENABLED" in compose_source:
    violations.append("Docker Compose must not expose a VLM enable/disable switch")
if 'VLM_BACKEND: "http_service"' not in compose_source:
    violations.append("Docker Compose must force the standalone PaddleOCR-VL HTTP backend")
if "condition: service_healthy" not in compose_source:
    violations.append("The application must wait for the mandatory PaddleOCR-VL service")
if 'READINESS_REQUIRE_VLM: "1"' not in compose_source:
    violations.append("Docker Compose readiness must require the PaddleOCR-VL service")
if "OLLAMA_MODEL=gemma4:latest" not in env_example_source:
    violations.append(".env.example must document the tested gemma4:latest baseline")
if "RAG_EMBEDDING_MODEL=embeddinggemma:latest" not in env_example_source:
    violations.append(".env.example must document the tested embeddinggemma:latest baseline")
if "vlm_enabled:" in config_source or "vlm_enabled:" in pipeline_source:
    violations.append("The extraction API must not expose a VLM enable/disable switch")
settings_source = SETTINGS_PATH.read_text(encoding="utf-8")
if "VLM_ENABLED" in settings_source:
    violations.append("Runtime settings must not expose a VLM enable/disable switch")
if 'VLM_BACKEND = "http_service"' not in settings_source:
    violations.append("Runtime settings must force the standalone PaddleOCR-VL backend")

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
