"""Inactive next-pipeline stage for final result and artifact publication."""

from __future__ import annotations

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.contracts.presentation import FinalizationRequest
from receipt_intelligence.extraction.presentation.artifacts import (
    CompatibilityFilesystemArtifactStore,
)
from receipt_intelligence.extraction.presentation.finalization import (
    CompatibilityFinalizationService,
)
from receipt_intelligence.extraction.services.finalization import ReceiptFinalizationService
from receipt_intelligence.extraction.state import ExtractionPhase, StageContractError


class NextFinalizationStage:
    name = "next_finalize"
    input_phase = ExtractionPhase.CATEGORIZED
    output_phase = ExtractionPhase.FINALIZED

    def __init__(self, service: ReceiptFinalizationService | None = None) -> None:
        self._service = service

    def run(self, context: ExtractionContext) -> ExtractionContext:
        correction = context.require_correction().result
        extraction = context.require_structured_extraction().result
        transcription = context.require_transcription().result
        finalization = context.require_finalized()
        categorization = finalization.next_categorization
        if correction is None or extraction is None or transcription is None or categorization is None:
            raise StageContractError(
                "NextFinalizationStage requires transcription, extraction, correction, and "
                "categorization results."
            )
        service = self._service or CompatibilityFinalizationService(
            artifact_store=CompatibilityFilesystemArtifactStore(context.config.result_dir),
            app_version=get_app_version(),
            overwrite=True,
        )
        context.emit(
            self.name,
            "running",
            "Publishing final receipt and compatibility artifacts.",
        )
        result = service.finalize(
            FinalizationRequest(
                run_id=context.config.run_id,
                receipt=correction.accepted_receipt,
                validation=correction.final_validation,
                categorization=categorization,
                stage_trace=tuple(context.stage_trace),
                upstream_metadata={
                    "transcription": {
                        "row_count": len(transcription.rows),
                        "crop_count": len(transcription.fragments),
                        "used_full_image_fallback": transcription.used_full_image_fallback,
                    },
                    "extraction": {
                        "scalar_task_count": len(extraction.scalar_results),
                        "missing_scalar_tasks": list(extraction.missing_scalar_tasks),
                        "item_status": (
                            extraction.item_result.status.value
                            if extraction.item_result is not None
                            else "skipped"
                        ),
                    },
                    "correction": {
                        "changed": correction.changed,
                        "corrected_codes": sorted(correction.corrected_codes),
                        "remaining_failed_codes": sorted(
                            correction.final_validation.failed_codes
                        ),
                        "attempt_count": len(correction.attempts),
                    },
                },
            )
        )
        finalization.next_finalization = result
        finalization.output_receipt = result.receipt
        finalization.categorized_receipt = result.receipt
        finalization.pipeline_meta = result.pipeline_metadata
        context.prepared.paths.update(result.paths)
        context.emit(
            self.name,
            "done",
            "Final receipt and compatibility artifacts published.",
            decision=result.validation.status,
            failed_codes=sorted(result.validation.failed_codes),
            artifact_count=len(result.artifacts),
        )
        return context


__all__ = ["NextFinalizationStage"]
