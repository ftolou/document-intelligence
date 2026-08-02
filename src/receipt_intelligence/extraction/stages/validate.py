"""Inactive next-pipeline stage for pure deterministic validation."""

from __future__ import annotations

from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.contracts.validation import ValidationRequest
from receipt_intelligence.extraction.state import ExtractionPhase, StageContractError


class ValidationStage:
    name = "next_validation"
    input_phase = ExtractionPhase.EXTRACTED
    output_phase = ExtractionPhase.VALIDATED

    def run(self, context: ExtractionContext) -> ExtractionContext:
        service = context.dependencies.validation_service
        if service is None:
            raise StageContractError(
                "ValidationStage requires ExtractionDependencies.validation_service."
            )
        extraction = context.require_structured_extraction().result
        if extraction is None:
            raise StageContractError("ValidationStage requires structured extraction result.")
        context.emit(
            self.name,
            "running",
            "Running read-only deterministic validation; no values will be changed.",
        )
        diagnostics = extraction.diagnostics
        report = service.validate(
            ValidationRequest(
                receipt=extraction.receipt,
                item_contract=extraction.item_contract,
                item_pipeline_enabled=bool(
                    diagnostics.get("item_pipeline_enabled", extraction.item_result is not None)
                ),
                selected_scalar_tasks=tuple(
                    diagnostics.get("selected_scalar_tasks") or ()
                ),
                money_tolerance=0.02,
                vat_rate_tolerance=0.02,
            )
        )
        context.begin_validation_stage().report = report
        context.emit(
            self.name,
            "done",
            "Deterministic validation completed without correction.",
            validation_status=report.status,
            failed_codes=sorted(report.failed_codes),
        )
        return context


__all__ = ["ValidationStage"]
