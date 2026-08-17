from receipt_intelligence.extraction.factory import build_extraction_workflow


def test_canonical_workflow_stage_order_is_explicit() -> None:
    assert [stage.name for stage in build_extraction_workflow().stages] == [
        "prepare",
        "transcription",
        "structured_extraction",
        "validation",
        "correction",
        "categorization",
        "finalize",
    ]
