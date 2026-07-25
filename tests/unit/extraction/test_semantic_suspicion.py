from __future__ import annotations

from receipt_intelligence.extraction.validation.semantic_suspicion import (
    attach_semantic_suspicion,
    evaluate_semantic_suspicion,
)


def test_large_quantity_without_unit_price_triggers_non_mutating_review() -> None:
    receipt = {
        "merchant": {"name": "Example"},
        "items": [
            {"description": "Bella", "quantity": 23, "unit_price": None, "line_total": 7.50},
            {"description": "Rustika", "quantity": 1, "line_total": 7.50},
        ],
        "totals": {"grand_total": 15.0},
    }
    original_items = [dict(item) for item in receipt["items"]]
    report = {
        "balanced": True,
        "import_decision": "import",
        "issues": [],
    }

    suspicion = evaluate_semantic_suspicion(receipt, report)
    augmented = attach_semantic_suspicion(report, suspicion)

    assert suspicion["triggered"] is True
    assert any(
        issue["code"] == "LARGE_QUANTITY_WITHOUT_UNIT_PRICE" for issue in suspicion["issues"]
    )
    assert augmented["import_decision"] == "needs_review"
    assert receipt["items"] == original_items


def test_normal_balanced_items_do_not_trigger_semantic_review() -> None:
    receipt = {
        "items": [
            {
                "description": "Milk",
                "product_description": "Milk",
                "quantity": 2,
                "unit_price": 1.25,
                "line_total": 2.50,
            }
        ]
    }
    report = {"balanced": True, "import_decision": "import", "issues": []}

    suspicion = evaluate_semantic_suspicion(receipt, report)

    assert suspicion["triggered"] is False
    assert suspicion["status"] == "clean"


def test_zero_value_item_is_diagnostic_but_not_a_standalone_trigger() -> None:
    receipt = {"items": [{"description": "Free sample", "quantity": 1, "line_total": 0.0}]}

    suspicion = evaluate_semantic_suspicion(receipt, {"issues": []})

    assert suspicion["triggered"] is False
    assert suspicion["low_count"] == 1
    assert suspicion["issues"][0]["code"] == "ZERO_VALUE_ITEM"
