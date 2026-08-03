"""Typed deterministic validation engine with standalone-report parity."""

from __future__ import annotations

from collections.abc import Sequence

from receipt_intelligence.extraction.contracts.validation import (
    ValidationCheck,
    ValidationReport,
    ValidationRequest,
    ValidationSeverity,
    ValidationStatus,
)
from receipt_intelligence.extraction.services.validation import ReceiptValidationService
from receipt_intelligence.extraction.validation.facts import ValidationFacts, money_float
from receipt_intelligence.extraction.validation.rules import (
    CurrencyRule,
    ItemRules,
    PaymentRule,
    ReceiptAmountRule,
    TotalRules,
    VatRules,
)
from receipt_intelligence.extraction.validation.rules.base import ValidationRule


class DeterministicValidationEngine(ReceiptValidationService):
    def __init__(self, rules: Sequence[ValidationRule] | None = None) -> None:
        self._rules = tuple(
            rules
            or (
                ReceiptAmountRule(),
                ItemRules(),
                TotalRules(),
                VatRules(),
                PaymentRule(),
                CurrencyRule(),
            )
        )

    def validate(self, request: ValidationRequest) -> ValidationReport:
        facts = ValidationFacts.build(request)
        checks: list[ValidationCheck] = []
        for rule in self._rules:
            checks.extend(rule.evaluate(facts))
        failed = [check for check in checks if check.status is ValidationStatus.FAILED]
        error_count = sum(check.severity is ValidationSeverity.ERROR for check in failed)
        review_count = sum(check.severity is ValidationSeverity.REVIEW for check in failed)
        status = "invalid" if error_count else ("review_required" if review_count else "valid")
        status_counts = {
            value.value: sum(check.status is value for check in checks)
            for value in (
                ValidationStatus.PASSED,
                ValidationStatus.FAILED,
                ValidationStatus.SKIPPED,
                ValidationStatus.OBSERVED,
            )
        }
        raw = {
            "status": status,
            "policy": {
                "money_tolerance": request.money_tolerance,
                "vat_rate_tolerance": request.vat_rate_tolerance,
                "uses_decimal_arithmetic": True,
                "changes_model_values": False,
                "correction_applied": False,
                "selected_scalar_tasks": list(request.selected_scalar_tasks),
            },
            "summary": {
                **status_counts,
                "error_count": error_count,
                "review_count": review_count,
            },
            "metrics": {
                "item_count": len(facts.items),
                "priced_item_count": len(facts.numeric_item_prices),
                "item_sum": money_float(facts.item_sum),
                "final_purchase_total": money_float(facts.final_total),
                "discount_total": money_float(facts.discount_total),
                "net_amount": money_float(facts.net_amount),
                "vat_amount": money_float(facts.vat_amount),
                "payment_received": money_float(facts.payment_received),
                "change_returned": money_float(facts.change_returned),
                "vat_line_count": len(facts.vat_lines),
            },
            "checks": [check.to_dict() for check in checks],
        }
        return ValidationReport(status=status, checks=tuple(checks), raw=raw)


__all__ = ["DeterministicValidationEngine"]
