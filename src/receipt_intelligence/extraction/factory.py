"""Composition root for the receipt extraction workflow."""

from receipt_intelligence.extraction.stages import (
    CategorizationStage,
    CorrectionStage,
    FinalizationStage,
    PreparationStage,
    StructuredExtractionStage,
    TranscriptionStage,
    ValidationStage,
)
from receipt_intelligence.extraction.workflow import ReceiptExtractionWorkflow


def build_extraction_workflow() -> ReceiptExtractionWorkflow:
    return ReceiptExtractionWorkflow(
        [
            PreparationStage(),
            TranscriptionStage(),
            StructuredExtractionStage(),
            ValidationStage(),
            CorrectionStage(),
            CategorizationStage(),
            FinalizationStage(),
        ]
    )


__all__ = ["build_extraction_workflow"]
