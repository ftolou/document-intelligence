from receipt_intelligence.extraction.factory import build_extraction_workflow


def test_validation_stage_is_active_in_the_canonical_workflow() -> None:
    workflow = build_extraction_workflow()
    stages = [stage.name for stage in workflow.stages]

    assert stages[3] == "validation"
