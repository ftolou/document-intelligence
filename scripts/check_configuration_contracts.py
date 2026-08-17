#!/usr/bin/env python3
"""Fail CI when the canonical image-first extraction contract regresses."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "receipt_intelligence"
violations: list[str] = []

config_source = (SRC / "extraction" / "config.py").read_text(encoding="utf-8")
pipeline_source = (SRC / "pipeline" / "integrated_receipt_pipeline.py").read_text(encoding="utf-8")
factory_source = (SRC / "extraction" / "factory.py").read_text(encoding="utf-8")
job_source = (SRC / "services" / "job_processing.py").read_text(encoding="utf-8")
settings_source = (SRC / "settings.py").read_text(encoding="utf-8")
env_source = (ROOT / ".env.example").read_text(encoding="utf-8")
compose_source = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

if "source_image_path: Path" not in config_source:
    violations.append("ExtractionConfig must require source_image_path")
for obsolete in (
    "ocr_json_path: Path",
    "vlm_service_url:",
    "spatial_canvas_width:",
    "gpu_orchestration:",
):
    if obsolete in config_source:
        violations.append(f"ExtractionConfig contains obsolete field: {obsolete}")
for forbidden in ("resolve_extraction_strategy", "ExtractionStrategy", "build_default"):
    if forbidden in pipeline_source:
        violations.append(f"Pipeline still contains strategy selection: {forbidden}")
for stage in (
    "PreparationStage",
    "TranscriptionStage",
    "StructuredExtractionStage",
    "ValidationStage",
    "CorrectionStage",
    "CategorizationStage",
    "FinalizationStage",
):
    if stage not in factory_source:
        violations.append(f"Canonical workflow is missing {stage}")
if "self.ocr_engine.recognize(" in job_source or "OcrRequest(" in job_source:
    violations.append("Job processing must not run preliminary full-image OCR")
if "ExtractionRequest(" not in job_source or "run_receipt_extraction(" not in job_source:
    violations.append("Job processing must use the typed extraction API")

for obsolete in ("EXTRACTION_STRATEGY=", "VLM_SERVICE_URL=", "VLM_BACKEND="):
    if obsolete in env_source:
        violations.append(f".env.example contains obsolete setting: {obsolete}")
for required in ("QWEN_TRANSCRIPTION_MODEL=", "EXTRACTION_MAX_CROPS=", "CORRECTION_ENABLED="):
    if required not in env_source:
        violations.append(f".env.example is missing setting: {required}")
for obsolete in ("receipt-vlm", "VLM_SERVICE_URL", "READINESS_REQUIRE_VLM"):
    if obsolete in compose_source or obsolete in settings_source:
        violations.append(f"Runtime still references removed VLM service: {obsolete}")

if violations:
    print("Configuration contract violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Canonical extraction configuration contracts passed.")
