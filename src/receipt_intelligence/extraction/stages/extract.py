"""Inactive next-pipeline stage for Gemma scalar/item extraction."""

from __future__ import annotations

from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.contracts.extraction import StructuredExtractionRequest
from receipt_intelligence.extraction.state import ExtractionPhase, StageContractError


class StructuredExtractionStage:
    name = "next_structured_extraction"
    input_phase = ExtractionPhase.TRANSCRIBED
    output_phase = ExtractionPhase.EXTRACTED

    def run(self, context: ExtractionContext) -> ExtractionContext:
        service = context.dependencies.structured_extraction_service
        if service is None:
            raise StageContractError(
                "StructuredExtractionStage requires "
                "ExtractionDependencies.structured_extraction_service."
            )
        transcription = context.require_transcription().result
        if transcription is None:
            raise StageContractError("StructuredExtractionStage requires transcription result.")
        context.emit(
            self.name,
            "running",
            "Running Gemma scalar specialists and direct item extraction.",
        )
        result = service.extract(
            StructuredExtractionRequest(
                run_id=context.config.run_id,
                transcription=transcription,
            )
        )
        context.begin_structured_extraction_stage().result = result
        context.emit(
            self.name,
            "done",
            "Gemma structured extraction completed without validation or correction.",
            scalar_task_count=len(result.scalar_results),
            item_status=(result.item_result.status.value if result.item_result else "skipped"),
        )
        return context


__all__ = ["StructuredExtractionStage"]
