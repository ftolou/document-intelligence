"""Application boundary for specialist, validator-gated correction."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.contracts.common import ReadonlyJsonObject
from receipt_intelligence.extraction.contracts.correction import CorrectionResult
from receipt_intelligence.extraction.contracts.transcription import TranscriptionResult
from receipt_intelligence.extraction.contracts.validation import ValidationReport


class ReceiptCorrectionService(Protocol):
    def correct(
        self,
        *,
        receipt: ReadonlyJsonObject,
        transcription: TranscriptionResult,
        validation: ValidationReport,
    ) -> CorrectionResult: ...


__all__ = ["ReceiptCorrectionService"]
