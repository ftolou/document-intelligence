"""Composition helper for the inactive next-pipeline transcription subsystem."""

from __future__ import annotations

from receipt_intelligence.adapters.multimodal import OllamaMultimodalGateway
from receipt_intelligence.adapters.text_detection import PaddleTextDetectionEngine
from receipt_intelligence.extraction.settings import PipelineSettings
from receipt_intelligence.extraction.transcription.service import (
    CanonicalReceiptTranscriptionService,
)
from receipt_intelligence.prompts.registry import default_prompt_registry


def build_canonical_transcription_service(
    settings: PipelineSettings,
    *,
    overwrite: bool = True,
) -> CanonicalReceiptTranscriptionService:
    return CanonicalReceiptTranscriptionService(
        detector=PaddleTextDetectionEngine(
            backend=settings.detection.backend,
            model_name=settings.detection.model_name,
        ),
        multimodal_gateway=OllamaMultimodalGateway(settings.transcription.ollama_url),
        prompt_registry=default_prompt_registry(),
        result_dir=settings.runtime.result_dir,
        detection_settings=settings.detection,
        crop_settings=settings.crop_planning,
        transcription_settings=settings.transcription,
        overwrite=overwrite,
    )


__all__ = ["build_canonical_transcription_service"]
