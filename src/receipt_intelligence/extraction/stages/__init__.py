"""Default receipt extraction stages."""

from receipt_intelligence.extraction.stages.finalize import FinalizationStage
from receipt_intelligence.extraction.stages.parse import MainParsingStage
from receipt_intelligence.extraction.stages.prepare import PreparationStage
from receipt_intelligence.extraction.stages.repair import RepairAndCorrectionStage
from receipt_intelligence.extraction.stages.visual import VisualEvidenceStage

__all__ = [
    "PreparationStage",
    "VisualEvidenceStage",
    "MainParsingStage",
    "RepairAndCorrectionStage",
    "FinalizationStage",
]
