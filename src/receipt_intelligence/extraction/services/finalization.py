"""Application boundary for final receipt publication and compatibility artifacts."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.contracts.presentation import (
    FinalizationRequest,
    FinalizationResult,
)


class ReceiptFinalizationService(Protocol):
    def finalize(self, request: FinalizationRequest) -> FinalizationResult: ...


__all__ = ["ReceiptFinalizationService"]
