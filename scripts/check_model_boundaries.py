#!/usr/bin/env python3
"""Fail CI when model-backed features bypass provider-neutral ports."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "receipt_intelligence"
violations: list[str] = []

for package in ("rag", "rag_sql"):
    for path in (SRC / package).rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in ("urllib.request", "requests.post", "OllamaTextResponse"):
            if token in source:
                violations.append(f"{path.relative_to(ROOT)} bypasses the model port: {token}")

required_files = (
    SRC / "application" / "ports" / "llm.py",
    SRC / "application" / "ports" / "multimodal.py",
    SRC / "application" / "ports" / "text_detection.py",
    SRC / "adapters" / "llm" / "ollama_gateway.py",
    SRC / "adapters" / "multimodal" / "ollama.py",
    SRC / "adapters" / "text_detection" / "paddle.py",
    SRC / "extraction" / "dependencies.py",
)
for required in required_files:
    if not required.exists():
        violations.append(f"Missing required boundary module: {required.relative_to(ROOT)}")

job_source = (SRC / "services" / "job_processing.py").read_text(encoding="utf-8")
if "paddleocr" in job_source.lower() or "OcrRequest" in job_source:
    violations.append("JobProcessingService must not invoke Paddle directly")

if violations:
    print("Model boundary violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Model, multimodal, and text-detection boundary checks passed.")
