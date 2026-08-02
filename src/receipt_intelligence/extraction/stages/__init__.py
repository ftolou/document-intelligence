"""Default receipt extraction stages."""

from receipt_intelligence.extraction.stages.correct import CorrectionStage
from receipt_intelligence.extraction.stages.extract import StructuredExtractionStage
from receipt_intelligence.extraction.stages.finalize import FinalizationStage
from receipt_intelligence.extraction.stages.overview import SpatialOverviewStage
from receipt_intelligence.extraction.stages.parse import MainParsingStage
from receipt_intelligence.extraction.stages.prepare import PreparationStage
from receipt_intelligence.extraction.stages.repair import RepairAndCorrectionStage
from receipt_intelligence.extraction.stages.transcribe import TranscriptionStage
from receipt_intelligence.extraction.stages.validate import ValidationStage
from receipt_intelligence.extraction.stages.visual import VisualEvidenceStage

__all__ = [
    "PreparationStage",
    "VisualEvidenceStage",
    "TranscriptionStage",
    "StructuredExtractionStage",
    "CorrectionStage",
    "ValidationStage",
    "SpatialOverviewStage",
    "MainParsingStage",
    "RepairAndCorrectionStage",
    "FinalizationStage",
]
