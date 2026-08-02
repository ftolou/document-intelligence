"""Application boundary for post-validation receipt categorization."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.contracts.presentation import (
    CategorizationRequest,
    CategorizationResult,
)


class ReceiptCategorizationService(Protocol):
    def categorize(self, request: CategorizationRequest) -> CategorizationResult: ...


__all__ = ["ReceiptCategorizationService"]
