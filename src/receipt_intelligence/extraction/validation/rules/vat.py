"""VAT-line rate, sum, and gross reconciliation checks."""

from decimal import Decimal

from receipt_intelligence.extraction.contracts.validation import (
    ValidationCheck,
    ValidationSeverity,
    ValidationStatus,
)
from receipt_intelligence.extraction.validation.facts import (
    MONEY_QUANTUM,
    ValidationFacts,
    money_float,
    request_money_close,
)


class VatRules:
    def evaluate(self, facts: ValidationFacts) -> tuple[ValidationCheck, ...]:
        checks: list[ValidationCheck] = []
        if not facts.vat_lines:
            checks.append(_check("VAT_LINE_RATE_ARITHMETIC", ValidationStatus.SKIPPED, "No VAT lines were extracted."))
        elif facts.vat_rate_failures:
            checks.append(_check(
                "VAT_LINE_RATE_ARITHMETIC",
                ValidationStatus.FAILED,
                "One or more VAT-line rate calculations are inconsistent.",
                severity=ValidationSeverity.REVIEW,
                details=list(facts.vat_rate_failures),
            ))
        else:
            checks.append(_check(
                "VAT_LINE_RATE_ARITHMETIC",
                ValidationStatus.PASSED,
                "All complete VAT lines satisfy net amount multiplied by tax rate within currency rounding tolerance.",
            ))

        if (
            facts.vat_lines
            and len(facts.line_vat_values) == len(facts.vat_lines)
            and facts.vat_amount is not None
        ):
            vat_line_sum = sum(facts.line_vat_values, Decimal("0.00")).quantize(MONEY_QUANTUM)
            if request_money_close(facts.request, vat_line_sum, facts.vat_amount):
                checks.append(_check(
                    "VAT_LINE_SUM_RECONCILIATION",
                    ValidationStatus.PASSED,
                    "The sum of VAT-line amounts equals the VAT total.",
                    values={
                        "vat_line_sum": money_float(vat_line_sum),
                        "vat_amount": money_float(facts.vat_amount),
                        "difference": money_float(vat_line_sum - facts.vat_amount),
                    },
                ))
            else:
                checks.append(_check(
                    "VAT_LINE_SUM_RECONCILIATION",
                    ValidationStatus.FAILED,
                    "The sum of VAT-line amounts does not equal the VAT total.",
                    severity=ValidationSeverity.REVIEW,
                    values={
                        "vat_line_sum": money_float(vat_line_sum),
                        "vat_amount": money_float(facts.vat_amount),
                        "difference": money_float(vat_line_sum - facts.vat_amount),
                    },
                ))
        else:
            checks.append(_check(
                "VAT_LINE_SUM_RECONCILIATION",
                ValidationStatus.SKIPPED,
                "A complete set of VAT-line amounts and a VAT total were not available.",
                details={
                    "vat_line_count": len(facts.vat_lines),
                    "complete_vat_amount_count": len(facts.line_vat_values),
                    "incomplete_vat_line_indices": list(facts.incomplete_vat_line_indices),
                },
            ))

        if (
            facts.vat_lines
            and len(facts.line_net_values) == len(facts.vat_lines)
            and len(facts.line_vat_values) == len(facts.vat_lines)
            and facts.final_total is not None
        ):
            vat_net_sum = sum(facts.line_net_values, Decimal("0.00")).quantize(MONEY_QUANTUM)
            vat_line_sum = sum(facts.line_vat_values, Decimal("0.00")).quantize(MONEY_QUANTUM)
            calculated_gross = vat_net_sum + vat_line_sum
            if request_money_close(facts.request, calculated_gross, facts.final_total):
                checks.append(_check(
                    "VAT_LINES_GROSS_RECONCILIATION",
                    ValidationStatus.PASSED,
                    "VAT-line net and VAT sums equal the final purchase total.",
                    values={
                        "vat_net_sum": money_float(vat_net_sum),
                        "vat_line_sum": money_float(vat_line_sum),
                        "calculated_gross": money_float(calculated_gross),
                        "final_purchase_total": money_float(facts.final_total),
                        "difference": money_float(calculated_gross - facts.final_total),
                    },
                ))
            else:
                checks.append(_check(
                    "VAT_LINES_GROSS_RECONCILIATION",
                    ValidationStatus.FAILED,
                    "VAT-line net and VAT sums do not equal the final purchase total.",
                    severity=ValidationSeverity.REVIEW,
                    values={
                        "vat_net_sum": money_float(vat_net_sum),
                        "vat_line_sum": money_float(vat_line_sum),
                        "calculated_gross": money_float(calculated_gross),
                        "final_purchase_total": money_float(facts.final_total),
                        "difference": money_float(calculated_gross - facts.final_total),
                    },
                ))
        else:
            checks.append(_check(
                "VAT_LINES_GROSS_RECONCILIATION",
                ValidationStatus.SKIPPED,
                "Complete VAT-line net and VAT amounts plus a final total were not available.",
            ))
        return tuple(checks)


def _check(
    code: str,
    status: ValidationStatus,
    message: str,
    *,
    severity: ValidationSeverity = ValidationSeverity.INFO,
    values: dict | None = None,
    details: object = None,
) -> ValidationCheck:
    return ValidationCheck(
        code=code,
        status=status,
        severity=severity,
        message=message,
        values=values or {},
        details=details,
    )


__all__ = ["VatRules"]
