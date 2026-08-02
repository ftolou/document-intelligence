from pathlib import Path


def test_validation_stage_remains_inactive() -> None:
    stage_source = Path(
        "src/receipt_intelligence/extraction/stages/validate.py"
    ).read_text(encoding="utf-8")
    assert 'name = "next_validation"' in stage_source
    assert "input_phase = ExtractionPhase.EXTRACTED" in stage_source
    assert "output_phase = ExtractionPhase.VALIDATED" in stage_source

    factory_path = Path("src/receipt_intelligence/extraction/factory.py")
    if factory_path.exists():
        factory_source = factory_path.read_text(encoding="utf-8")
        assert "ValidationStage" not in factory_source
        assert "next_validation" not in factory_source
