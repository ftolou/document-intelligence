from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from receipt_intelligence.application.ports.multimodal import (
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)
from receipt_intelligence.application.ports.text_detection import (
    DetectedTextRegion,
    TextDetectionRequest,
    TextDetectionResult,
)
from receipt_intelligence.extraction.contracts.transcription import (
    TranscriptionFragment,
    TranscriptionRequest,
)
from receipt_intelligence.extraction.settings import (
    CropPlanningSettings,
    DetectionSettings,
    TranscriptionSettings,
)
from receipt_intelligence.extraction.transcription.canonical import (
    build_canonical_rows,
    serialize_canonical_rows,
)
from receipt_intelligence.extraction.transcription.crop_planner import (
    determine_effective_crop_count,
)
from receipt_intelligence.extraction.transcription.service import (
    CanonicalReceiptTranscriptionService,
)
from receipt_intelligence.prompts.registry import default_prompt_registry


class FakeDetector:
    def __init__(self, result: TextDetectionResult) -> None:
        self.result = result
        self.requests: list[TextDetectionRequest] = []

    def detect(self, request: TextDetectionRequest) -> TextDetectionResult:
        self.requests.append(request)
        return self.result


class FakeGateway:
    def __init__(self, responses: dict[str, str | Exception]) -> None:
        self.responses = responses
        self.requests: list[MultimodalGenerationRequest] = []

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        self.requests.append(request)
        crop_name = request.image_paths[0].stem.removeprefix("group_")
        value = self.responses[crop_name]
        if isinstance(value, Exception):
            raise value
        return MultimodalGenerationResult(text=value, text_source="message.content")


def _region(index: int, y: float) -> DetectedTextRegion:
    return DetectedTextRegion(
        region_id=f"D{index:04d}",
        polygon=((10.0, y), (90.0, y), (90.0, y + 10.0), (10.0, y + 10.0)),
        score=0.99,
    )


def test_canonical_rows_are_ordered_and_transport_wrappers_are_removed() -> None:
    fragments = (
        TranscriptionFragment(crop_id="G002", text="- SUMME 3,00", order=1),
        TranscriptionFragment(
            crop_id="G001",
            text="```text\nR0007 :: ARTIKEL 3,00\n```",
            order=0,
        ),
    )

    rows = build_canonical_rows(fragments)

    assert [row.text for row in rows] == ["ARTIKEL 3,00", "SUMME 3,00"]
    assert serialize_canonical_rows(rows) == (
        "BEGIN_RECEIPT\nR0001 :: ARTIKEL 3,00\nR0002 :: SUMME 3,00\nEND_RECEIPT"
    )


def test_crop_count_uses_rows_and_height_width_aspect_ratio() -> None:
    count, metadata = determine_effective_crop_count(
        requested_crops=4,
        detected_line_count=12,
        image_width=100,
        image_height=600,
        target_rows_per_crop=18,
        single_crop_max_rows=25,
        single_crop_max_aspect_ratio=2.0,
    )

    assert count == 3
    assert metadata["image_aspect_ratio_h_over_w"] == 6.0


def test_detector_failure_uses_whole_image_without_post_validation(tmp_path: Path) -> None:
    image_path = tmp_path / "receipt.png"
    Image.new("RGB", (100, 200), "white").save(image_path)
    detector = FakeDetector(TextDetectionResult(regions=(), image_width=100, image_height=200))
    gateway = FakeGateway({"GFULL": "SHOP\nSUMME 5,00"})
    service = CanonicalReceiptTranscriptionService(
        detector=detector,
        multimodal_gateway=gateway,
        prompt_registry=default_prompt_registry(),
        result_dir=tmp_path,
        detection_settings=DetectionSettings(minimum_lines=3),
        crop_settings=CropPlanningSettings(max_crops=4),
        transcription_settings=TranscriptionSettings(
            ollama_url="http://ollama",
            model="qwen3.5:4b",
        ),
    )

    result = service.transcribe(TranscriptionRequest(image_path, "run-1"))

    assert result.diagnostics["crop_plan"]["status"] == "fallback_full_image"
    assert [row.text for row in result.rows] == ["SHOP", "SUMME 5,00"]
    assert len(gateway.requests) == 1
    assert result.diagnostics["post_transcription_validation_used"] is False


def test_failed_crop_discards_partial_output_and_retries_whole_image(tmp_path: Path) -> None:
    image_path = tmp_path / "receipt.png"
    Image.new("RGB", (100, 600), "white").save(image_path)
    regions = tuple(_region(index, 20.0 + index * 40.0) for index in range(12))
    detector = FakeDetector(TextDetectionResult(regions=regions, image_width=100, image_height=600))
    gateway = FakeGateway(
        {
            "G001": "PARTIAL SHOULD BE DISCARDED",
            "G002": RuntimeError("transport failure"),
            "GFULL_RUNTIME": "FULL RECEIPT\nSUMME 7,00",
        }
    )
    service = CanonicalReceiptTranscriptionService(
        detector=detector,
        multimodal_gateway=gateway,
        prompt_registry=default_prompt_registry(),
        result_dir=tmp_path,
        detection_settings=DetectionSettings(minimum_lines=3),
        crop_settings=CropPlanningSettings(
            max_crops=2,
            target_rows_per_crop=4,
            single_crop_max_rows=3,
            single_crop_max_aspect_ratio=2.0,
        ),
        transcription_settings=TranscriptionSettings(
            ollama_url="http://ollama",
            model="qwen3.5:4b",
            retries=0,
        ),
    )

    result = service.transcribe(TranscriptionRequest(image_path, "run-2"))

    assert [row.text for row in result.rows] == ["FULL RECEIPT", "SUMME 7,00"]
    assert "PARTIAL SHOULD BE DISCARDED" not in result.canonical_text
    assert result.diagnostics["runtime_full_image_fallback"] is True
    called = [request.image_paths[0].stem for request in gateway.requests]
    assert called == ["group_G001", "group_G002", "group_GFULL_RUNTIME"]


def test_empty_qwen_text_is_a_transport_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "receipt.png"
    Image.new("RGB", (100, 200), "white").save(image_path)
    detector = FakeDetector(TextDetectionResult(regions=(), image_width=100, image_height=200))

    class EmptyGateway:
        def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
            raise RuntimeError("empty response")

    service = CanonicalReceiptTranscriptionService(
        detector=detector,
        multimodal_gateway=EmptyGateway(),
        prompt_registry=default_prompt_registry(),
        result_dir=tmp_path,
        detection_settings=DetectionSettings(),
        crop_settings=CropPlanningSettings(),
        transcription_settings=TranscriptionSettings(
            ollama_url="http://ollama",
            model="qwen3.5:4b",
            retries=0,
        ),
    )

    with pytest.raises(RuntimeError, match="no nonempty transcription"):
        service.transcribe(TranscriptionRequest(image_path, "run-3"))
