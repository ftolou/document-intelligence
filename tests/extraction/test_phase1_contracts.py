from __future__ import annotations

from pathlib import Path

import pytest

from receipt_intelligence.extraction.config import ExtractionConfig
from receipt_intelligence.extraction.contracts import (
    CanonicalTranscriptionRow,
    ReceiptCrop,
    TranscriptionFragment,
    TranscriptionResult,
    ValidationReport,
)
from receipt_intelligence.extraction.settings import PipelineSettings


def test_grouped_settings_bridge_keeps_parsing_and_transcription_models_separate(
    tmp_path: Path,
) -> None:
    config = ExtractionConfig(
        ocr_json_path=tmp_path / "ocr.json",
        source_image_path=tmp_path / "receipt.jpg",
        result_dir=tmp_path / "results",
        run_id="run-1",
        ollama_url="http://localhost:11434",
        model="gemma4",
    )

    settings = PipelineSettings.from_extraction_config(
        config,
        transcription_model="qwen3.5:4b",
    )

    assert settings.transcription.model == "qwen3.5:4b"
    assert settings.parsing.model == "gemma4"
    assert settings.runtime.run_id == "run-1"
    assert settings.source_image_path == tmp_path / "receipt.jpg"


def test_grouped_settings_bridge_requires_source_image(tmp_path: Path) -> None:
    config = ExtractionConfig(
        ocr_json_path=tmp_path / "ocr.json",
        source_image_path=None,
        result_dir=tmp_path / "results",
        run_id="run-1",
        ollama_url="http://localhost:11434",
        model="gemma4",
    )

    with pytest.raises(ValueError, match="source_image_path"):
        PipelineSettings.from_extraction_config(
            config,
            transcription_model="qwen3.5:4b",
        )


def test_transcription_contract_preserves_one_canonical_representation(
    tmp_path: Path,
) -> None:
    crop = ReceiptCrop(
        crop_id="crop-001",
        image_path=tmp_path / "crop.png",
        source_box=(0.0, 0.0, 100.0, 200.0),
        order=0,
    )
    fragment = TranscriptionFragment(
        crop_id=crop.crop_id,
        text="SUMME 12,34",
        order=0,
    )
    row = CanonicalTranscriptionRow(
        row_id="R0001",
        text="SUMME 12,34",
        source_crop_ids=(crop.crop_id,),
    )

    result = TranscriptionResult(
        canonical_text="R0001 :: SUMME 12,34",
        rows=(row,),
        crops=(crop,),
        fragments=(fragment,),
    )

    assert result.rows[0].row_id == "R0001"
    assert result.canonical_text.startswith("R0001")


def test_validation_report_adapts_current_dictionary_contract() -> None:
    report = ValidationReport.from_legacy(
        {
            "status": "review_required",
            "checks": [
                {
                    "code": "ITEM_SUM_RECONCILIATION",
                    "status": "failed",
                    "severity": "error",
                    "message": "Item sum does not reconcile.",
                    "details": {"difference": "1.00"},
                }
            ],
        }
    )

    assert report.failed_codes == frozenset({"ITEM_SUM_RECONCILIATION"})
    assert report.find("ITEM_SUM_RECONCILIATION") is not None
