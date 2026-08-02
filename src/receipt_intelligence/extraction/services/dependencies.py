"""Service bundle consumed by the future extraction workflow composition root."""

from __future__ import annotations

from dataclasses import dataclass

from receipt_intelligence.extraction.services.correction import ReceiptCorrectionService
from receipt_intelligence.extraction.services.structured_extraction import (
    StructuredExtractionService,
)
from receipt_intelligence.extraction.services.transcription import TranscriptionService
from receipt_intelligence.extraction.services.validation import ReceiptValidationService


@dataclass(frozen=True, slots=True)
class PipelineServices:
    transcription: TranscriptionService
    structured_extraction: StructuredExtractionService
    validation: ReceiptValidationService
    correction: ReceiptCorrectionService


__all__ = ["PipelineServices"]
