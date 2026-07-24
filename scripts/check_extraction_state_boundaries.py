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
    errors.append("ExtractionContext must not expose the string-based require(attribute) API.")

for path in sorted(STAGES.glob("*.py")):
    text = path.read_text(encoding="utf-8")
    if ".require(" in text:
        errors.append(f"{path.relative_to(ROOT)} still uses string-based context.require().")

required_artifacts = {
    "PreparedArtifacts",
    "VisualArtifacts",
    "OverviewArtifacts",
    "ParsingArtifacts",
    "RepairArtifacts",
    "FinalizationArtifacts",
}
state_tree = ast.parse((EXTRACTION / "state.py").read_text(encoding="utf-8"))
defined_classes = {
    node.name for node in state_tree.body if isinstance(node, ast.ClassDef)
}
missing = sorted(required_artifacts - defined_classes)
if missing:
    errors.append(f"Missing typed extraction artifact classes: {', '.join(missing)}")

expected_stages = {
    "prepare.py": ("ExtractionPhase.CREATED", "ExtractionPhase.PREPARED"),
    "visual.py": ("ExtractionPhase.PREPARED", "ExtractionPhase.VISUAL_READY"),
    "overview.py": ("ExtractionPhase.VISUAL_READY", "ExtractionPhase.OVERVIEW_READY"),
    "parse.py": ("ExtractionPhase.OVERVIEW_READY", "ExtractionPhase.PARSED"),
    "repair.py": ("ExtractionPhase.PARSED", "ExtractionPhase.REPAIRED"),
    "finalize.py": ("ExtractionPhase.REPAIRED", "ExtractionPhase.FINALIZED"),
}
for filename, (input_phase, output_phase) in expected_stages.items():
    text = (STAGES / filename).read_text(encoding="utf-8")
    if f"input_phase = {input_phase}" not in text:
        errors.append(f"{filename} does not declare input phase {input_phase}.")
    if f"output_phase = {output_phase}" not in text:
        errors.append(f"{filename} does not declare output phase {output_phase}.")

legacy_fields = {
    "visual_result",
    "visual_evidence",
    "llm_result",
    "receipt",
    "ocr_context",
    "report",
    "final_receipt",
    "final_report",
    "categorized_receipt",
}
context_tree = ast.parse(context_text)
for node in ast.walk(context_tree):
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.target.id in legacy_fields:
            errors.append(
                f"ExtractionContext still stores ungrouped stage field {node.target.id!r}."
            )

if errors:
    for error in errors:
        print(f"ERROR: {error}")
    raise SystemExit(1)

print("Extraction state boundary checks passed.")
