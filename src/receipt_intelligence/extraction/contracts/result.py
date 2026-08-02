"""Public typed result returned by the future extraction application service."""

from __future__ import annotations

from dataclasses import dataclass

from receipt_intelligence.extraction.contracts.common import JsonObject, StageArtifact
from receipt_intelligence.extraction.contracts.correction import CorrectionResult
from receipt_intelligence.extraction.contracts.extraction import StructuredExtractionResult
from receipt_intelligence.extraction.contracts.presentation import CategorizationResult
from receipt_intelligence.extraction.contracts.transcription import TranscriptionResult
from receipt_intelligence.extraction.contracts.validation import ValidationReport


@dataclass(frozen=True, slots=True)
class ReceiptPipelineResult:
    receipt: JsonObject
    validation: ValidationReport
    transcription: TranscriptionResult
    extraction: StructuredExtractionResult
    correction: CorrectionResult
    pipeline_metadata: JsonObject
    artifacts: tuple[StageArtifact, ...] = ()
    categorization: CategorizationResult | None = None


__all__ = ["ReceiptPipelineResult"]
