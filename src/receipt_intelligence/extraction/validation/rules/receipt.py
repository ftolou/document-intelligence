"""Receipt-level monetary domain checks."""

from receipt_intelligence.extraction.contracts.validation import (
    ValidationCheck,
    ValidationSeverity,
    ValidationStatus,
)
from receipt_intelligence.extraction.validation.facts import ValidationFacts, money_float


class ReceiptAmountRule:
    def evaluate(self, facts: ValidationFacts) -> tuple[ValidationCheck, ...]:
        money_values = {
            "final_purchase_total": facts.final_total,
            "pre_discount_total": facts.pre_discount_total,
            "net_amount": facts.net_amount,
            "discount_total": facts.discount_total,
            "payment_received": facts.payment_received,
            "change_returned": facts.change_returned,
            "vat_amount": facts.vat_amount,
        }
        negative_values = {
            name: money_float(value)
            for name, value in money_values.items()
            if value is not None and value < 0
        }
        if negative_values:
            return (
                ValidationCheck(
                    code="NONNEGATIVE_RECEIPT_AMOUNTS",
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message="One or more receipt-level monetary values are negative.",
                    values=negative_values,
                ),
            )
        return (
            ValidationCheck(
                code="NONNEGATIVE_RECEIPT_AMOUNTS",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message="All available receipt-level monetary values are nonnegative.",
            ),
        )


__all__ = ["ReceiptAmountRule"]
