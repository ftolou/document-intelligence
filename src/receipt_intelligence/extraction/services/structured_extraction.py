"""Application boundary for Gemma scalar/item extraction and receipt assembly."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.contracts.extraction import StructuredExtractionResult
from receipt_intelligence.extraction.contracts.transcription import TranscriptionResult


class StructuredExtractionService(Protocol):
    def extract(self, transcription: TranscriptionResult) -> StructuredExtractionResult: ...


__all__ = ["StructuredExtractionService"]
