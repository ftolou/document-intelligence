"""Receipt total, VAT, and payment reconciliation checks."""

from receipt_intelligence.extraction.contracts.validation import (
    ValidationCheck,
    ValidationSeverity,
    ValidationStatus,
)
from receipt_intelligence.extraction.validation.facts import (
    ValidationFacts,
    money_float,
    request_money_close,
)


class TotalRules:
    def evaluate(self, facts: ValidationFacts) -> tuple[ValidationCheck, ...]:
        checks: list[ValidationCheck] = []
        request = facts.request
        if (
            facts.pre_discount_total is not None
            and facts.discount_total is not None
            and facts.final_total is not None
        ):
            expected_final = facts.pre_discount_total - facts.discount_total
            if request_money_close(request, expected_final, facts.final_total):
                checks.append(_check(
                    "PRE_DISCOUNT_TOTAL_RECONCILIATION",
                    ValidationStatus.PASSED,
                    "Pre-discount total minus discount total equals the final purchase total.",
                    values={
                        "pre_discount_total": money_float(facts.pre_discount_total),
                        "discount_total": money_float(facts.discount_total),
                        "final_purchase_total": money_float(facts.final_total),
                        "difference": money_float(expected_final - facts.final_total),
                    },
                ))
            else:
                checks.append(_check(
                    "PRE_DISCOUNT_TOTAL_RECONCILIATION",
                    ValidationStatus.FAILED,
                    "Pre-discount total minus discount total does not equal the final purchase total.",
                    severity=ValidationSeverity.REVIEW,
                    values={
                        "pre_discount_total": money_float(facts.pre_discount_total),
                        "discount_total": money_float(facts.discount_total),
                        "expected_final_purchase_total": money_float(expected_final),
                        "final_purchase_total": money_float(facts.final_total),
                        "difference": money_float(expected_final - facts.final_total),
                    },
                ))
        else:
            checks.append(_check(
                "PRE_DISCOUNT_TOTAL_RECONCILIATION",
                ValidationStatus.SKIPPED,
                "Pre-discount total, discount total, and final total were not all available.",
            ))

        if facts.net_amount is not None and facts.vat_amount is not None and facts.final_total is not None:
            calculated_gross = facts.net_amount + facts.vat_amount
            if request_money_close(request, calculated_gross, facts.final_total):
                checks.append(_check(
                    "NET_PLUS_VAT_RECONCILIATION",
                    ValidationStatus.PASSED,
                    "Net amount plus VAT equals the final purchase total.",
                    values={
                        "net_amount": money_float(facts.net_amount),
                        "vat_amount": money_float(facts.vat_amount),
                        "final_purchase_total": money_float(facts.final_total),
                        "difference": money_float(calculated_gross - facts.final_total),
                    },
                ))
            else:
                checks.append(_check(
                    "NET_PLUS_VAT_RECONCILIATION",
                    ValidationStatus.FAILED,
                    "Net amount plus VAT does not equal the final purchase total.",
                    severity=ValidationSeverity.REVIEW,
                    values={
                        "net_amount": money_float(facts.net_amount),
                        "vat_amount": money_float(facts.vat_amount),
                        "calculated_gross": money_float(calculated_gross),
                        "final_purchase_total": money_float(facts.final_total),
                        "difference": money_float(calculated_gross - facts.final_total),
                    },
                ))
        else:
            checks.append(_check(
                "NET_PLUS_VAT_RECONCILIATION",
                ValidationStatus.SKIPPED,
                "Net amount, VAT amount, and final total were not all available.",
            ))
        return tuple(checks)


class PaymentRule:
    def evaluate(self, facts: ValidationFacts) -> tuple[ValidationCheck, ...]:
        if (
            facts.payment_received is not None
            and facts.change_returned is not None
            and facts.final_total is not None
        ):
            calculated_total = facts.payment_received - facts.change_returned
            if request_money_close(facts.request, calculated_total, facts.final_total):
                return (_check(
                    "PAYMENT_CHANGE_RECONCILIATION",
                    ValidationStatus.PASSED,
                    "Payment received minus change returned equals the final purchase total.",
                    values={
                        "payment_received": money_float(facts.payment_received),
                        "change_returned": money_float(facts.change_returned),
                        "final_purchase_total": money_float(facts.final_total),
                        "difference": money_float(calculated_total - facts.final_total),
                    },
                ),)
            return (_check(
                "PAYMENT_CHANGE_RECONCILIATION",
                ValidationStatus.FAILED,
                "Payment received minus change returned does not equal the final purchase total.",
                severity=ValidationSeverity.REVIEW,
                values={
                    "payment_received": money_float(facts.payment_received),
                    "change_returned": money_float(facts.change_returned),
                    "calculated_purchase_total": money_float(calculated_total),
                    "final_purchase_total": money_float(facts.final_total),
                    "difference": money_float(calculated_total - facts.final_total),
                },
            ),)
        return (_check(
            "PAYMENT_CHANGE_RECONCILIATION",
            ValidationStatus.SKIPPED,
            "Payment received, change returned, and final total were not all available.",
        ),)


def _check(
    code: str,
    status: ValidationStatus,
    message: str,
    *,
    severity: ValidationSeverity = ValidationSeverity.INFO,
    values: dict | None = None,
) -> ValidationCheck:
    return ValidationCheck(
        code=code,
        status=status,
        severity=severity,
        message=message,
        values=values or {},
    )


__all__ = ["PaymentRule", "TotalRules"]
