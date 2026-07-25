#!/usr/bin/env python3
"""Fail CI when feature packages bypass model and vision ports."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "receipt_intelligence"
violations: list[str] = []


def scan_forbidden(package: str, forbidden: tuple[str, ...]) -> None:
    for path in (SRC / package).rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                violations.append(
                    f"{path.relative_to(ROOT)} contains forbidden dependency {token!r}"
                )


scan_forbidden(
    "rag",
    (
        "receipt_intelligence.extraction.parsing.llm_parser",
        "receipt_intelligence.observability.ollama",
        "OllamaTextResponse",
        "OllamaCallMetrics",
    ),
)
scan_forbidden(
    "rag_sql",
    (
        "receipt_intelligence.extraction.parsing.llm_parser",
        "receipt_intelligence.observability.ollama",
        "OllamaTextResponse",
        "OllamaCallMetrics",
    ),
)
scan_forbidden(
    "extraction/stages",
    (
        "receipt_intelligence.engines.vl_engine",
        "receipt_intelligence.services.ollama_control",
    ),
)

required_files = (
    SRC / "application" / "ports" / "llm.py",
    SRC / "application" / "ports" / "ocr.py",
    SRC / "application" / "ports" / "vlm.py",
    SRC / "application" / "ports" / "model_lifecycle.py",
    SRC / "adapters" / "llm" / "ollama_gateway.py",
    SRC / "extraction" / "dependencies.py",
)
for required in required_files:
    if not required.exists():
        violations.append(f"Missing required boundary module: {required.relative_to(ROOT)}")

llm_port_path = SRC / "application" / "ports" / "llm.py"
llm_tree = ast.parse(llm_port_path.read_text(encoding="utf-8"), filename=str(llm_port_path))
for node in llm_tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "GenerationResult":
        if node.bases:
            violations.append("GenerationResult must be a normal result object, not a subtype")
        break
else:
    violations.append("GenerationResult is missing from the LLM port")

parser_source = (SRC / "extraction" / "parsing" / "llm_parser.py").read_text(encoding="utf-8")
if "class OllamaTextResponse" in parser_source:
    violations.append("The string-subclass model response must not be reintroduced")

vlm_port_source = (SRC / "application" / "ports" / "vlm.py").read_text(encoding="utf-8")
for infrastructure_field in ("backend_name", "service_url", "trusted_command"):
    if infrastructure_field in vlm_port_source:
        violations.append(
            f"VlmRequest must not expose infrastructure field {infrastructure_field!r}"
        )

ocr_port_source = (SRC / "application" / "ports" / "ocr.py").read_text(encoding="utf-8")
for provider_field in ("use_angle_cls", "det_limit_side_len"):
    if provider_field in ocr_port_source:
        violations.append(f"OcrRequest must not expose Paddle-specific field {provider_field!r}")

job_source = (SRC / "services" / "job_processing.py").read_text(encoding="utf-8")
if "from receipt_intelligence.engines.ocr_engine" in job_source:
    violations.append("JobProcessingService must depend on OcrEngine, not PaddleOCR directly")
if "self.ocr_engine.recognize(" not in job_source:
    violations.append("JobProcessingService must invoke the injected OCR port")

if violations:
    print("Model and vision boundary violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Model, OCR, VLM, and lifecycle boundary checks passed.")
