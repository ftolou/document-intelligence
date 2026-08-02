"""Inactive next-pipeline stage for specialist validator-gated correction."""

from __future__ import annotations

from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.contracts.correction import CorrectionRequest
from receipt_intelligence.extraction.state import ExtractionPhase, StageContractError


class CorrectionStage:
    name = "next_correction"
    input_phase = ExtractionPhase.VALIDATED
    output_phase = ExtractionPhase.CORRECTED

    def run(self, context: ExtractionContext) -> ExtractionContext:
        service = context.dependencies.correction_service
        if service is None:
            raise StageContractError(
                "CorrectionStage requires ExtractionDependencies.correction_service."
            )
        extraction = context.require_structured_extraction().result
        validation = context.require_validation().report
        transcription = context.require_transcription().result
        if extraction is None or validation is None or transcription is None:
            raise StageContractError(
                "CorrectionStage requires transcription, structured extraction, and validation."
            )
        diagnostics = extraction.diagnostics
        context.emit(
            self.name,
            "running",
            "Running source-evidence specialists with deterministic acceptance gating.",
        )
        result = service.correct(
            CorrectionRequest(
                run_id=context.config.run_id,
                receipt=extraction.receipt,
                transcription=transcription,
                validation=validation,
                item_contract=extraction.item_contract,
                item_pipeline_enabled=bool(
                    diagnostics.get("item_pipeline_enabled", extraction.item_result is not None)
                ),
                selected_scalar_tasks=tuple(diagnostics.get("selected_scalar_tasks") or ()),
            )
        )
        context.begin_correction_stage().result = result
        context.emit(
            self.name,
            "done",
            "Specialist correction completed; unsupported failures remain open.",
            changed=result.changed,
            corrected_codes=sorted(result.corrected_codes),
            remaining_failed_codes=sorted(result.final_validation.failed_codes),
        )
        return context


__all__ = ["CorrectionStage"]
