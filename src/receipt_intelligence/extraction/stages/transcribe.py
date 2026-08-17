"""Canonical Paddle/Qwen transcription stage."""

from __future__ import annotations

from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.contracts.transcription import TranscriptionRequest
from receipt_intelligence.extraction.state import ExtractionPhase, StageContractError


class TranscriptionStage:
    name = "transcription"
    input_phase = ExtractionPhase.PREPARED
    output_phase = ExtractionPhase.TRANSCRIBED

    def run(self, context: ExtractionContext) -> ExtractionContext:
        service = context.dependencies.transcription_service
        if service is None:
            raise StageContractError(
                "TranscriptionStage requires ExtractionDependencies.transcription_service."
            )
        source_image = context.config.source_image_path
        context.emit(
            self.name,
            "running",
            "Running Paddle geometry and Qwen canonical transcription.",
        )
        result = service.transcribe(
            TranscriptionRequest(
                source_image_path=source_image,
                run_id=context.config.run_id,
            )
        )
        context.begin_transcription_stage().result = result
        context.register_artifacts(result.artifacts)
        context.emit(
            self.name,
            "done",
            "Canonical transcription completed without post-transcription semantic validation.",
            row_count=len(result.rows),
            crop_count=len(result.crops),
        )
        return context


__all__ = ["TranscriptionStage"]
