"""Composition root for the default extraction workflow."""

from receipt_intelligence.extraction.stages import (
    FinalizationStage,
    MainParsingStage,
    PreparationStage,
    RepairAndCorrectionStage,
    VisualEvidenceStage,
)
from receipt_intelligence.extraction.workflow import ReceiptExtractionWorkflow


def build_default_extraction_workflow() -> ReceiptExtractionWorkflow:
    return ReceiptExtractionWorkflow(
        [
            PreparationStage(),
            VisualEvidenceStage(),
            MainParsingStage(),
            RepairAndCorrectionStage(),
            FinalizationStage(),
        ]
    )
