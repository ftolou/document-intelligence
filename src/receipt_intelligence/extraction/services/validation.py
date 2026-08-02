"""Application boundary for read-only deterministic receipt validation."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.contracts.validation import (
    ValidationReport,
    ValidationRequest,
)


class ReceiptValidationService(Protocol):
    def validate(self, request: ValidationRequest) -> ValidationReport: ...


__all__ = ["ReceiptValidationService"]
