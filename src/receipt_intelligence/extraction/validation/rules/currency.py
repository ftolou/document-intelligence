"""Currency consistency check."""

from receipt_intelligence.extraction.contracts.validation import (
    ValidationCheck,
    ValidationSeverity,
    ValidationStatus,
)
from receipt_intelligence.extraction.validation.facts import ValidationFacts


class CurrencyRule:
    def evaluate(self, facts: ValidationFacts) -> tuple[ValidationCheck, ...]:
        currencies = sorted({source["currency"] for source in facts.currency_sources})
        details = [dict(source) for source in facts.currency_sources]
        if len(currencies) > 1:
            return (
                ValidationCheck(
                    code="CURRENCY_CONSISTENCY",
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.REVIEW,
                    message="Receipt-level specialists returned conflicting currencies.",
                    values={"currencies": currencies},
                    details=details,
                ),
            )
        if currencies:
            return (
                ValidationCheck(
                    code="CURRENCY_CONSISTENCY",
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.INFO,
                    message="All available receipt-level currencies are consistent.",
                    values={"currency": currencies[0]},
                    details=details,
                ),
            )
        return (
            ValidationCheck(
                code="CURRENCY_CONSISTENCY",
                status=ValidationStatus.SKIPPED,
                severity=ValidationSeverity.INFO,
                message="No usable currency value was available.",
            ),
        )


__all__ = ["CurrencyRule"]
