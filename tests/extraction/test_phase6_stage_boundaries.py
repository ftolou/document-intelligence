from __future__ import annotations

import ast
from pathlib import Path


def _class_assignments(path: Path, class_name: str) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values: dict[str, str] = {}
            for item in node.body:
                if not isinstance(item, ast.Assign) or len(item.targets) != 1:
                    continue
                target = item.targets[0]
                if isinstance(target, ast.Name) and isinstance(item.value, ast.Attribute):
                    values[target.id] = item.value.attr
            return values
    raise AssertionError(f"Class {class_name} not found in {path}")


def test_categorization_and_finalization_form_canonical_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    categorize = _class_assignments(
        root / "src/receipt_intelligence/extraction/stages/categorize.py",
        "CategorizationStage",
    )
    finalize = _class_assignments(
        root / "src/receipt_intelligence/extraction/stages/finalize.py",
        "FinalizationStage",
    )
    assert categorize["input_phase"] == "CORRECTED"
    assert categorize["output_phase"] == "CATEGORIZED"
    assert finalize["input_phase"] == "CATEGORIZED"
    assert finalize["output_phase"] == "FINALIZED"
