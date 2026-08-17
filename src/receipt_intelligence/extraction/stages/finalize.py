"""Publish the final receipt and stable application artifacts."""

from __future__ import annotations

from collections.abc import Iterable

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.contracts.common import JsonObject
from receipt_intelligence.extraction.contracts.presentation import FinalizationRequest
from receipt_intelligence.extraction.presentation.artifacts import (
    CompatibilityFilesystemArtifactStore,
)
from receipt_intelligence.extraction.presentation.finalization import (
    CompatibilityFinalizationService,
)
from receipt_intelligence.extraction.services.finalization import ReceiptFinalizationService
from receipt_intelligence.extraction.state import ExtractionPhase, StageContractError


class FinalizationStage:
    name = "finalize"
    input_phase = ExtractionPhase.CATEGORIZED
    output_phase = ExtractionPhase.FINALIZED

    def __init__(self, service: ReceiptFinalizationService | None = None) -> None:
        self._service = service

    def run(self, context: ExtractionContext) -> ExtractionContext:
        correction = context.require_correction().result
        extraction = context.require_structured_extraction().result
        transcription = context.require_transcription().result
        finalization = context.require_finalized()
        categorization = finalization.categorization
        if (
            correction is None
            or extraction is None
            or transcription is None
            or categorization is None
        ):
            raise StageContractError(
                "FinalizationStage requires transcription, extraction, correction, and "
                "categorization results."
            )
        service = self._service or CompatibilityFinalizationService(
            artifact_store=CompatibilityFilesystemArtifactStore(context.config.result_dir),
            app_version=get_app_version(),
            overwrite=True,
        )
        context.emit(self.name, "running", "Publishing final receipt and stable artifacts.")
        result = service.finalize(
            FinalizationRequest(
                run_id=context.config.run_id,
                receipt=correction.accepted_receipt,
                validation=correction.final_validation,
                categorization=categorization,
                stage_trace=_completed_finalization_trace(context.stage_trace),
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
                        "remaining_failed_codes": sorted(correction.final_validation.failed_codes),
                        "attempt_count": len(correction.attempts),
                    },
                },
            )
        )
        finalization.result = result
        context.prepared.paths.update(result.paths)
        context.emit(
            self.name,
            "done",
            "Final receipt and stable artifacts published.",
            decision=result.validation.status,
            failed_codes=sorted(result.validation.failed_codes),
            artifact_count=len(result.artifacts),
        )
        return context


def _completed_finalization_trace(
    stage_trace: Iterable[JsonObject],
) -> tuple[JsonObject, ...]:
    snapshot = [dict(entry) for entry in stage_trace if isinstance(entry, dict)]
    for entry in reversed(snapshot):
        if entry.get("stage") == FinalizationStage.name:
            if entry.get("status") == "running":
                entry["status"] = "done"
            break
    return tuple(snapshot)


__all__ = ["FinalizationStage"]
