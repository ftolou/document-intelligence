"""Composition root for the opt-in Qwen/Gemma extraction workflow."""

from receipt_intelligence.extraction.stages.categorize import CategorizationStage
from receipt_intelligence.extraction.stages.correct import CorrectionStage
from receipt_intelligence.extraction.stages.extract import StructuredExtractionStage
from receipt_intelligence.extraction.stages.next_prepare import NextPreparationStage
from receipt_intelligence.extraction.stages.publish import NextFinalizationStage
from receipt_intelligence.extraction.stages.transcribe import TranscriptionStage
from receipt_intelligence.extraction.stages.validate import ValidationStage
from receipt_intelligence.extraction.workflow import ReceiptExtractionWorkflow


def build_next_extraction_workflow() -> ReceiptExtractionWorkflow:
    return ReceiptExtractionWorkflow(
        [
            NextPreparationStage(),
            TranscriptionStage(),
            StructuredExtractionStage(),
            ValidationStage(),
            CorrectionStage(),
            CategorizationStage(),
            NextFinalizationStage(),
        ]
    )


__all__ = ["build_next_extraction_workflow"]
