"""Application boundary for read-only deterministic receipt validation."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.contracts.common import ReadonlyJsonObject
from receipt_intelligence.extraction.contracts.transcription import TranscriptionResult
from receipt_intelligence.extraction.contracts.validation import ValidationReport


class ReceiptValidationService(Protocol):
    def validate(
        self,
        receipt: ReadonlyJsonObject,
        transcription: TranscriptionResult,
    ) -> ValidationReport: ...


__all__ = ["ReceiptValidationService"]
