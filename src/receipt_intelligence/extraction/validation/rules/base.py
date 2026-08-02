"""Protocol for one deterministic validation rule group."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.contracts.validation import ValidationCheck
from receipt_intelligence.extraction.validation.facts import ValidationFacts


class ValidationRule(Protocol):
    def evaluate(self, facts: ValidationFacts) -> tuple[ValidationCheck, ...]: ...


__all__ = ["ValidationRule"]
