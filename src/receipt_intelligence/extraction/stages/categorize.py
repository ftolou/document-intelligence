"""Optional post-validation categorization stage."""

from __future__ import annotations

from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.contracts.presentation import CategorizationRequest
from receipt_intelligence.extraction.presentation.categorization import (
    ReceiptCategorizationAdapter,
)
from receipt_intelligence.extraction.services.categorization import ReceiptCategorizationService
from receipt_intelligence.extraction.state import (
    ExtractionPhase,
    FinalizationArtifacts,
    StageContractError,
)


class CategorizationStage:
    name = "categorization"
    input_phase = ExtractionPhase.CORRECTED
    output_phase = ExtractionPhase.CATEGORIZED

    def __init__(self, service: ReceiptCategorizationService | None = None) -> None:
        self._service = service

    def run(self, context: ExtractionContext) -> ExtractionContext:
        correction = context.require_correction().result
        if correction is None:
            raise StageContractError("CategorizationStage requires correction result.")
        service = self._service or ReceiptCategorizationAdapter(
            llm_gateway=context.dependencies.llm_gateway,
            ollama_url=context.config.ollama_url,
            model=context.config.categorization_model or context.config.model,
            num_ctx=context.config.categorization_num_ctx,
            num_predict=context.config.categorization_num_predict,
            keep_alive=context.config.keep_alive,
            timeout_seconds=context.config.categorization_timeout_seconds,
            format_json=context.config.categorization_format_json,
        )
        context.emit(
            self.name,
            "running" if context.config.categorization_enabled else "skipped",
            "Categorizing final corrected items without changing receipt arithmetic.",
        )
        result = service.categorize(
            CategorizationRequest(
                run_id=context.config.run_id,
                receipt=correction.accepted_receipt,
                enabled=context.config.categorization_enabled,
            )
        )
        if context.finalized is None:
            context.finalized = FinalizationArtifacts()
        context.finalized.categorization = result
        context.emit(
            self.name,
            "done" if result.status.value not in {"error"} else "error",
            "Post-validation categorization finished.",
            categorization_status=result.status.value,
            warning_count=len(result.warnings),
            error=result.error,
        )
        return context


__all__ = ["CategorizationStage"]
