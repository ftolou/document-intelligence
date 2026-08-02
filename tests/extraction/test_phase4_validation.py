from __future__ import annotations

import copy

from receipt_intelligence.extraction.contracts.validation import (
    ValidationRequest,
    ValidationSeverity,
    ValidationStatus,
)
from receipt_intelligence.extraction.validation.engine import DeterministicValidationEngine


def _receipt() -> dict:
    return {
        "receipt_metadata": {"currency": "EUR"},
        "items": [
            {
                "name": "A",
                "quantity": None,
                "unit": None,
                "final_price": 5.0,
                "original_price": None,
                "discount_amount": None,
            },
            {
                "name": "B",
                "quantity": None,
                "unit": None,
                "final_price": 3.0,
                "original_price": None,
                "discount_amount": None,
            },
        ],
        "totals": {
            "final_purchase_total": {"final_purchase_total": 8.0, "currency": "EUR"},
            "pre_discount_total": None,
            "net_amount": {"net_amount": 6.72, "currency": "EUR"},
        },
        "discount": {"discount_total": None},
        "payment": {"payment_received": None, "change_returned": None},
        "tax": {
            "vat_amount": {"vat_amount": 1.28, "currency": "EUR"},
            "vat_lines": [
                {"rate_percent": 19.0, "net_amount": 6.72, "vat_amount": 1.28}
            ],
        },
    }


def _request(receipt: dict, item_contract: dict | None = None) -> ValidationRequest:
    return ValidationRequest(
        receipt=receipt,
        item_contract=item_contract or {"status": "valid", "errors": [], "warnings": []},
        item_pipeline_enabled=True,
        selected_scalar_tasks=("final_purchase_total", "vat_amount", "vat_lines"),
    )


def test_valid_receipt_and_report_shape() -> None:
    report = DeterministicValidationEngine().validate(_request(_receipt()))
    assert report.status == "valid"
    assert report.failed_codes == frozenset()
    raw = report.to_dict()
    assert raw["policy"]["changes_model_values"] is False
    assert raw["policy"]["money_tolerance"] == 0.02
    assert [check["code"] for check in raw["checks"]] == [
        "NONNEGATIVE_RECEIPT_AMOUNTS",
        "ITEM_CONTRACT",
        "ITEMS_PRESENT",
        "ITEM_PRICES_COMPLETE",
        "ITEM_DISCOUNT_ARITHMETIC",
        "ITEM_SUM_RECONCILIATION",
        "PRE_DISCOUNT_TOTAL_RECONCILIATION",
        "NET_PLUS_VAT_RECONCILIATION",
        "VAT_LINE_RATE_ARITHMETIC",
        "VAT_LINE_SUM_RECONCILIATION",
        "VAT_LINES_GROSS_RECONCILIATION",
        "PAYMENT_CHANGE_RECONCILIATION",
        "CURRENCY_CONSISTENCY",
    ]


def test_missing_price_is_review_and_item_sum_is_skipped() -> None:
    receipt = _receipt()
    receipt["items"][1]["final_price"] = None
    contract = {
        "status": "valid_with_warnings",
        "errors": [],
        "warnings": [{"code": "MISSING_FINAL_PRICE", "location": "items[1].final_price"}],
    }
    report = DeterministicValidationEngine().validate(_request(receipt, contract))
    assert report.status == "review_required"
    assert {"ITEM_CONTRACT", "ITEM_PRICES_COMPLETE"} <= report.failed_codes
    assert report.find("ITEM_SUM_RECONCILIATION").status is ValidationStatus.SKIPPED


def test_vat_semantic_role_error_is_detected_without_mutation() -> None:
    receipt = _receipt()
    receipt["tax"]["vat_lines"] = [
        {"rate_percent": 19.0, "net_amount": 30.16, "vat_amount": 4.82}
    ]
    receipt["tax"]["vat_amount"] = {"vat_amount": 4.82, "currency": "EUR"}
    before = copy.deepcopy(receipt)
    report = DeterministicValidationEngine().validate(_request(receipt))
    assert "VAT_LINE_RATE_ARITHMETIC" in report.failed_codes
    assert "VAT_LINES_GROSS_RECONCILIATION" in report.failed_codes
    assert receipt == before


def test_structural_item_error_makes_receipt_invalid() -> None:
    contract = {
        "status": "invalid",
        "errors": [{"code": "NEGATIVE_DISCOUNT_AMOUNT"}],
        "warnings": [],
    }
    report = DeterministicValidationEngine().validate(_request(_receipt(), contract))
    assert report.status == "invalid"
    check = report.find("ITEM_CONTRACT")
    assert check is not None
    assert check.severity is ValidationSeverity.ERROR
