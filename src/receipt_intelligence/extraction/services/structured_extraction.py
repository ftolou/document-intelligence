"""Application boundary for Gemma scalar/item extraction and receipt assembly."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.contracts.extraction import (
    StructuredExtractionRequest,
    StructuredExtractionResult,
)


class StructuredExtractionService(Protocol):
    def extract(self, request: StructuredExtractionRequest) -> StructuredExtractionResult: ...


__all__ = ["StructuredExtractionService"]
