"""Inactive next-pipeline stage for canonical Paddle/Qwen transcription."""

from __future__ import annotations

from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.contracts.transcription import TranscriptionRequest
from receipt_intelligence.extraction.state import ExtractionPhase, StageContractError


class TranscriptionStage:
    """Produce canonical transcription without changing the active workflow factory."""

    name = "next_transcription"
    input_phase = ExtractionPhase.PREPARED
    output_phase = ExtractionPhase.TRANSCRIBED

    def run(self, context: ExtractionContext) -> ExtractionContext:
        service = context.dependencies.transcription_service
        if service is None:
            raise StageContractError(
                "TranscriptionStage requires ExtractionDependencies.transcription_service."
            )
        source_image = context.config.source_image_path
        if source_image is None:
            raise StageContractError("TranscriptionStage requires source_image_path.")
        context.emit(
            "next_transcription",
            "running",
            "Running Paddle geometry and Qwen canonical transcription.",
        )
        result = service.transcribe(
            TranscriptionRequest(
                source_image_path=source_image,
                run_id=context.config.run_id,
                legacy_ocr_json_path=context.config.ocr_json_path,
            )
        )
        context.begin_transcription_stage().result = result
        context.emit(
            "next_transcription",
            "done",
            "Canonical transcription completed without post-transcription semantic validation.",
            row_count=len(result.rows),
            crop_count=len(result.crops),
        )
        return context


__all__ = ["TranscriptionStage"]
