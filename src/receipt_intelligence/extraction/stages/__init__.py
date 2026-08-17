"""Stages in the canonical receipt extraction workflow."""

from receipt_intelligence.extraction.stages.categorize import CategorizationStage
from receipt_intelligence.extraction.stages.correct import CorrectionStage
from receipt_intelligence.extraction.stages.extract import StructuredExtractionStage
from receipt_intelligence.extraction.stages.finalize import FinalizationStage
from receipt_intelligence.extraction.stages.prepare import PreparationStage
from receipt_intelligence.extraction.stages.transcribe import TranscriptionStage
from receipt_intelligence.extraction.stages.validate import ValidationStage

__all__ = [
    "PreparationStage",
    "TranscriptionStage",
    "StructuredExtractionStage",
    "ValidationStage",
    "CorrectionStage",
    "CategorizationStage",
    "FinalizationStage",
]
