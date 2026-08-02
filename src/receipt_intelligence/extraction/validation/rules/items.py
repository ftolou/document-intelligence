"""Independent item-contract, completeness, arithmetic and sum checks."""

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


class ItemRules:
    def evaluate(self, facts: ValidationFacts) -> tuple[ValidationCheck, ...]:
        checks: list[ValidationCheck] = []
        request = facts.request
        contract = request.item_contract
        contract_status = contract.get("status")
        if not request.item_pipeline_enabled:
            checks.append(_check("ITEM_CONTRACT", ValidationStatus.SKIPPED, "Item extraction was disabled."))
        elif contract_status == "invalid":
            checks.append(_check(
                "ITEM_CONTRACT",
                ValidationStatus.FAILED,
                "The direct item output failed structural contract validation.",
                severity=ValidationSeverity.ERROR,
                details=contract.get("errors") or [],
            ))
        elif contract_status in {"valid", "valid_with_warnings"}:
            warnings = contract.get("warnings") or []
            if warnings:
                checks.append(_check(
                    "ITEM_CONTRACT",
                    ValidationStatus.FAILED,
                    "The direct item output is structurally valid but incomplete or ambiguous.",
                    severity=ValidationSeverity.REVIEW,
                    details=warnings,
                ))
            else:
                checks.append(_check(
                    "ITEM_CONTRACT",
                    ValidationStatus.PASSED,
                    "The direct item output satisfies the structural contract.",
                ))
        else:
            checks.append(_check(
                "ITEM_CONTRACT",
                ValidationStatus.FAILED,
                "No usable direct-item contract-validation result is available.",
                severity=ValidationSeverity.ERROR,
            ))

        if not request.item_pipeline_enabled:
            checks.append(_check("ITEMS_PRESENT", ValidationStatus.SKIPPED, "Item extraction was disabled."))
        elif not isinstance(facts.items_value, list):
            checks.append(_check(
                "ITEMS_PRESENT",
                ValidationStatus.FAILED,
                "The assembled receipt does not contain an item array.",
                severity=ValidationSeverity.ERROR,
            ))
        elif not facts.items:
            checks.append(_check(
                "ITEMS_PRESENT",
                ValidationStatus.FAILED,
                "No purchased items were extracted.",
                severity=ValidationSeverity.REVIEW,
                values={"item_count": 0},
            ))
        else:
            checks.append(_check(
                "ITEMS_PRESENT",
                ValidationStatus.PASSED,
                "At least one purchased item was extracted.",
                values={"item_count": len(facts.items)},
            ))

        if not request.item_pipeline_enabled:
            checks.append(_check("ITEM_PRICES_COMPLETE", ValidationStatus.SKIPPED, "Item extraction was disabled."))
        elif facts.missing_price_indices:
            checks.append(_check(
                "ITEM_PRICES_COMPLETE",
                ValidationStatus.FAILED,
                "One or more extracted items have no final price.",
                severity=ValidationSeverity.REVIEW,
                values={
                    "item_count": len(facts.items),
                    "priced_item_count": len(facts.numeric_item_prices),
                    "missing_price_count": len(facts.missing_price_indices),
                },
                details={"missing_item_indices": list(facts.missing_price_indices)},
            ))
        else:
            checks.append(_check(
                "ITEM_PRICES_COMPLETE",
                ValidationStatus.PASSED,
                "Every extracted item has a numeric final price.",
                values={"priced_item_count": len(facts.numeric_item_prices)},
            ))

        if facts.item_discount_failures:
            checks.append(_check(
                "ITEM_DISCOUNT_ARITHMETIC",
                ValidationStatus.FAILED,
                "One or more item-level original-price, discount, and final-price fields are inconsistent.",
                severity=ValidationSeverity.REVIEW,
                details=list(facts.item_discount_failures),
            ))
        elif request.item_pipeline_enabled and facts.items:
            checks.append(_check(
                "ITEM_DISCOUNT_ARITHMETIC",
                ValidationStatus.PASSED,
                "All verifiable item-level discount relationships are arithmetically consistent.",
            ))
        else:
            checks.append(_check(
                "ITEM_DISCOUNT_ARITHMETIC",
                ValidationStatus.SKIPPED,
                "No item-level discount relationship was available to validate.",
            ))

        if facts.duplicate_groups:
            checks.append(_check(
                "REPEATED_IDENTICAL_ITEM_OBJECTS",
                ValidationStatus.OBSERVED,
                "Identical item objects occur more than once. This is informational because repeated purchases are valid.",
                details={"item_index_groups": [list(value) for value in facts.duplicate_groups]},
            ))

        all_item_prices_available = bool(facts.items) and (
            len(facts.numeric_item_prices) == len(facts.items)
        )
        if not request.item_pipeline_enabled:
            checks.append(_check("ITEM_SUM_RECONCILIATION", ValidationStatus.SKIPPED, "Item extraction was disabled."))
        elif not all_item_prices_available:
            checks.append(_check(
                "ITEM_SUM_RECONCILIATION",
                ValidationStatus.SKIPPED,
                "The item sum cannot be validated until every item has a final price.",
                values={
                    "item_sum_partial": money_float(facts.item_sum),
                    "priced_item_count": len(facts.numeric_item_prices),
                    "item_count": len(facts.items),
                },
            ))
        elif facts.final_total is None:
            checks.append(_check(
                "ITEM_SUM_RECONCILIATION",
                ValidationStatus.SKIPPED,
                "No final purchase total is available for item-sum validation.",
                values={"item_sum": money_float(facts.item_sum)},
            ))
        else:
            assert facts.item_sum is not None
            direct_difference = facts.item_sum - facts.final_total
            discounted_difference = (
                facts.item_sum - facts.discount_total - facts.final_total
                if facts.discount_total is not None
                else None
            )
            if request_money_close(request, facts.item_sum, facts.final_total):
                checks.append(_check(
                    "ITEM_SUM_RECONCILIATION",
                    ValidationStatus.PASSED,
                    "The sum of item final prices equals the final purchase total.",
                    values={
                        "mode": "item_final_prices_include_all_discounts",
                        "item_sum": money_float(facts.item_sum),
                        "final_purchase_total": money_float(facts.final_total),
                        "difference": money_float(direct_difference),
                    },
                ))
            elif facts.discount_total is not None and request_money_close(
                request, facts.item_sum - facts.discount_total, facts.final_total
            ):
                checks.append(_check(
                    "ITEM_SUM_RECONCILIATION",
                    ValidationStatus.PASSED,
                    "The item sum reconciles after applying the explicit receipt-level discount.",
                    values={
                        "mode": "receipt_level_discount_not_allocated_to_items",
                        "item_sum": money_float(facts.item_sum),
                        "discount_total": money_float(facts.discount_total),
                        "final_purchase_total": money_float(facts.final_total),
                        "difference": money_float(discounted_difference),
                    },
                ))
            else:
                difference_equals_vat = (
                    facts.vat_amount is not None
                    and request_money_close(request, abs(direct_difference), facts.vat_amount)
                )
                message = (
                    "The item sum does not reconcile with the selected final purchase total. "
                    "The absolute difference equals the extracted VAT amount."
                    if difference_equals_vat
                    else "The item sum cannot be reconciled with the final purchase total, "
                    "with or without the explicit receipt-level discount."
                )
                checks.append(_check(
                    "ITEM_SUM_RECONCILIATION",
                    ValidationStatus.FAILED,
                    message,
                    severity=ValidationSeverity.REVIEW,
                    values={
                        "item_sum": money_float(facts.item_sum),
                        "final_purchase_total": money_float(facts.final_total),
                        "discount_total": money_float(facts.discount_total),
                        "vat_amount": money_float(facts.vat_amount),
                        "direct_difference": money_float(direct_difference),
                        "absolute_direct_difference": money_float(abs(direct_difference)),
                        "difference_after_discount": money_float(discounted_difference),
                        "item_total_difference_equals_vat": difference_equals_vat,
                    },
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


__all__ = ["ItemRules"]
