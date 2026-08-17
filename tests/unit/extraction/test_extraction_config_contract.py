from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from receipt_intelligence.extraction import ExtractionConfig, ExtractionRequest
from receipt_intelligence.pipeline.integrated_receipt_pipeline import (
    run_integrated_receipt_pipeline,
)


def required_values(tmp_path: Path) -> dict[str, object]:
    return {
        "source_image_path": tmp_path / "receipt.jpg",
        "result_dir": tmp_path,
        "run_id": "receipt-1",
        "ollama_url": "http://ollama:11434",
        "model": "gemma",
    }


def test_extraction_request_is_an_immutable_image_first_contract(tmp_path: Path) -> None:
    request = ExtractionRequest(**required_values(tmp_path))

    assert isinstance(request, ExtractionConfig)
    assert request.source_image_path == tmp_path / "receipt.jpg"
    assert request.max_crops == 4
    with pytest.raises(FrozenInstanceError):
        request.model = "other"  # type: ignore[misc]


def test_legacy_runtime_fields_are_absent() -> None:
    field_names = {field.name for field in fields(ExtractionRequest)}

    assert "ocr_json_path" not in field_names
    assert "vlm_service_url" not in field_names
    assert "spatial_canvas_width" not in field_names


def test_max_crops_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_crops"):
        ExtractionRequest(**required_values(tmp_path), max_crops=0)


def test_compatibility_wrapper_rejects_unknown_keywords(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="typoed_option"):
        run_integrated_receipt_pipeline(
            **required_values(tmp_path),
            typoed_option=True,
        )


def test_compatibility_wrapper_requires_an_image(tmp_path: Path) -> None:
    values = required_values(tmp_path)
    values.pop("source_image_path")
    with pytest.raises(TypeError, match="source_image_path"):
        run_integrated_receipt_pipeline(**values)
