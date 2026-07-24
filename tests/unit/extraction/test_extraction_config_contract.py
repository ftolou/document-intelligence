from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from receipt_intelligence.extraction import ExtractionConfig, ExtractionRequest
from receipt_intelligence.extraction.compatibility import (
    extraction_request_from_mapping,
    normalize_extraction_arguments,
)
from receipt_intelligence.pipeline.integrated_receipt_pipeline import (
    run_integrated_receipt_pipeline,
)


def required_values(tmp_path: Path) -> dict[str, object]:
    return {
        "ocr_json_path": tmp_path / "ocr.json",
        "result_dir": tmp_path,
        "run_id": "receipt-1",
        "ollama_url": "http://ollama:11434",
        "model": "gemma",
    }


def test_extraction_request_is_an_immutable_config_contract(tmp_path: Path) -> None:
    request = ExtractionRequest(**required_values(tmp_path))

    assert isinstance(request, ExtractionConfig)
    assert request.ocr_json_path == tmp_path / "ocr.json"
    assert request.gpu_orchestration == "none"

    with pytest.raises(FrozenInstanceError):
        request.model = "other"  # type: ignore[misc]


def test_legacy_gpu_names_map_to_canonical_fields(tmp_path: Path) -> None:
    request = extraction_request_from_mapping(
        {
            **required_values(tmp_path),
            "vlm_gpu_orchestration": "Sequential",
            "ollama_unload_before_vlm": True,
            "ollama_reload_after_vlm": True,
        }
    )

    assert request.gpu_orchestration == "sequential"
    assert request.unload_llm_before_vlm is True
    assert request.reload_llm_after_vlm is True


def test_unknown_options_are_rejected() -> None:
    with pytest.raises(TypeError, match="Unsupported extraction option"):
        normalize_extraction_arguments({"typoed_option": True})


def test_alias_and_canonical_name_cannot_be_supplied_together() -> None:
    with pytest.raises(TypeError, match="supplied more than once"):
        normalize_extraction_arguments(
            {
                "gpu_orchestration": "none",
                "vlm_gpu_orchestration": "sequential",
            }
        )


def test_compatibility_entry_point_rejects_unknown_keywords_before_execution(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="typoed_option"):
        run_integrated_receipt_pipeline(
            **required_values(tmp_path),
            typoed_option=True,
        )
