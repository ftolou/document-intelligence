from __future__ import annotations

from pathlib import Path

import pytest

from receipt_intelligence.application.ports.multimodal import (
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)
from receipt_intelligence.application.ports.text_detection import (
    DetectedTextRegion,
    TextDetectionRequest,
    TextDetectionResult,
)


def test_multimodal_request_requires_images() -> None:
    with pytest.raises(ValueError, match="image_paths"):
        MultimodalGenerationRequest(
            model="qwen3.5:4b",
            prompt="Transcribe the receipt.",
            image_paths=(),
        )


def test_multimodal_request_supports_thinking_and_schema_independently(
    tmp_path: Path,
) -> None:
    request = MultimodalGenerationRequest(
        model="qwen3.5:4b",
        prompt="Transcribe the receipt.",
        image_paths=(tmp_path / "receipt.jpg",),
        think=True,
        format_json=True,
        response_json_schema={"type": "object"},
    )

    assert request.think is True
    assert request.response_json_schema == {"type": "object"}
    assert MultimodalGenerationResult(text="receipt text").text == "receipt text"


def test_text_detection_contract_is_provider_neutral(tmp_path: Path) -> None:
    request = TextDetectionRequest(image_path=tmp_path / "receipt.jpg")
    region = DetectedTextRegion(
        region_id="region-1",
        polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
        score=0.95,
    )
    result = TextDetectionResult(
        regions=(region,),
        image_width=100,
        image_height=200,
    )

    assert request.language == "german"
    assert result.regions[0].region_id == "region-1"
