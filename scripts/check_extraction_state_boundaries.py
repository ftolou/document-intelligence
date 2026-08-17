#!/usr/bin/env python3
"""Enforce typed state boundaries between extraction workflow stages."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACTION = ROOT / "src" / "receipt_intelligence" / "extraction"
STAGES = EXTRACTION / "stages"
errors: list[str] = []

context_text = (EXTRACTION / "context.py").read_text(encoding="utf-8")
if "def require(" in context_text:
    errors.append("ExtractionContext must not expose string-based require(attribute).")

required_artifacts = {
    "PreparedArtifacts",
    "TranscriptionArtifacts",
    "StructuredExtractionArtifacts",
    "ValidationArtifacts",
    "CorrectionArtifacts",
    "FinalizationArtifacts",
}
state_tree = ast.parse((EXTRACTION / "state.py").read_text(encoding="utf-8"))
defined_classes = {node.name for node in state_tree.body if isinstance(node, ast.ClassDef)}
missing = sorted(required_artifacts - defined_classes)
if missing:
    errors.append(f"Missing typed extraction artifact classes: {', '.join(missing)}")

expected_stages = {
    "prepare.py": ("ExtractionPhase.CREATED", "ExtractionPhase.PREPARED"),
    "transcribe.py": ("ExtractionPhase.PREPARED", "ExtractionPhase.TRANSCRIBED"),
    "extract.py": ("ExtractionPhase.TRANSCRIBED", "ExtractionPhase.EXTRACTED"),
    "validate.py": ("ExtractionPhase.EXTRACTED", "ExtractionPhase.VALIDATED"),
    "correct.py": ("ExtractionPhase.VALIDATED", "ExtractionPhase.CORRECTED"),
    "categorize.py": ("ExtractionPhase.CORRECTED", "ExtractionPhase.CATEGORIZED"),
    "finalize.py": ("ExtractionPhase.CATEGORIZED", "ExtractionPhase.FINALIZED"),
}
for filename, (input_phase, output_phase) in expected_stages.items():
    text = (STAGES / filename).read_text(encoding="utf-8")
    if f"input_phase = {input_phase}" not in text:
        errors.append(f"{filename} does not declare input phase {input_phase}.")
    if f"output_phase = {output_phase}" not in text:
        errors.append(f"{filename} does not declare output phase {output_phase}.")

for obsolete in ("VisualArtifacts", "OverviewArtifacts", "ParsingArtifacts", "RepairArtifacts"):
    if obsolete in defined_classes:
        errors.append(f"Legacy extraction artifact remains: {obsolete}")

if errors:
    for error in errors:
        print(f"ERROR: {error}")
    raise SystemExit(1)

print("Extraction state boundary checks passed.")
