"""Application boundary for specialist, validator-gated correction."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.contracts.correction import CorrectionRequest, CorrectionResult


class ReceiptCorrectionService(Protocol):
    def correct(self, request: CorrectionRequest) -> CorrectionResult: ...


__all__ = ["ReceiptCorrectionService"]
