from __future__ import annotations

import ast
from pathlib import Path


def test_next_workflow_stage_order_is_explicit() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "receipt_intelligence"
        / "extraction"
        / "next_factory.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_next_extraction_workflow"
    )
    returned = next(node.value for node in function.body if isinstance(node, ast.Return))
    assert isinstance(returned, ast.Call)
    stage_list = returned.args[0]
    assert isinstance(stage_list, ast.List)
    assert [
        element.func.id
        for element in stage_list.elts
        if isinstance(element, ast.Call) and isinstance(element.func, ast.Name)
    ] == [
        "NextPreparationStage",
        "TranscriptionStage",
        "StructuredExtractionStage",
        "ValidationStage",
        "CorrectionStage",
        "CategorizationStage",
        "NextFinalizationStage",
    ]
