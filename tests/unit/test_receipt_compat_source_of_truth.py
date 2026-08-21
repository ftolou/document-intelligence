from __future__ import annotations

from receipt_intelligence.receipt_compat import (
    item_description,
    item_line_total,
    receipt_change,
    receipt_currency,
    receipt_date,
    receipt_grand_total,
    receipt_paid_total,
    receipt_payment_method,
    receipt_subtotal,
    receipt_tax_total,
    to_review_document,
)


def _next_receipt() -> dict:
    return {
        "receipt_metadata": {
            "date": None,
            "currency": None,
        },
        "date": "1999-01-01",
        "currency": "USD",
        "totals": {
            "net_amount": {"net_amount": None, "currency": "EUR"},
            "subtotal": 99.0,
            "final_purchase_total": {
                "final_purchase_total": 12.0,
                "currency": "EUR",
            },
            "grand_total": 99.0,
            "paid_total": 99.0,
            "tax_total": 99.0,
            "change": 7.5,
        },
        "tax": {
            "vat_amount": {"vat_amount": None, "currency": "EUR"},
        },
        "payment": {
            "payment_method": None,
            "payment_received": {"payment_received": None, "currency": "EUR"},
            "change_returned": {"change_returned": None, "currency": "EUR"},
        },
        "payments": [{"method": "cash", "amount": 99.0}],
        "items": [
            {
                "name": "Canonical item",
                "product_description": "STALE ITEM",
                "normalized_name": "stale normalized",
                "description": "STALE ITEM",
                "final_price": None,
                "line_total": 99.0,
                "total": 99.0,
                "amount": 99.0,
            }
        ],
    }


def test_explicit_next_schema_values_win_even_when_null() -> None:
    receipt = _next_receipt()

    assert receipt_date(receipt) is None
    assert receipt_currency(receipt) is None
    assert receipt_subtotal(receipt) is None
    assert receipt_tax_total(receipt) is None
    assert receipt_grand_total(receipt) == 12.0
    assert receipt_paid_total(receipt) is None
    assert receipt_change(receipt) is None
    assert receipt_payment_method(receipt) is None
    assert item_description(receipt["items"][0]) == "Canonical item"
    assert item_line_total(receipt["items"][0]) is None

    review = to_review_document(receipt)
    assert review["currency"] is None
    assert review["totals"]["subtotal"] is None
    assert review["totals"]["tax_total"] is None
    assert review["totals"]["grand_total"] == 12.0
    assert review["totals"]["paid_total"] is None
    assert review["totals"]["change"] is None
    assert review["payment_method"] is None
    assert review["items"][0]["description"] == "Canonical item"
    assert review["items"][0]["product_description"] == "Canonical item"
    assert review["items"][0]["line_total"] is None


def test_canonical_change_and_name_beat_stale_aliases() -> None:
    receipt = _next_receipt()
    receipt["payment"]["change_returned"] = {
        "change_returned": 1.25,
        "currency": "EUR",
    }
    item = receipt["items"][0]
    item["name"] = "NEW"
    item["product_description"] = "OLD"
    item["normalized_name"] = "OLD NORMALIZED"
    item["description"] = "OLD"

    assert receipt_change(receipt) == 1.25
    assert item_description(item) == "NEW"

    review = to_review_document(receipt)
    assert review["totals"]["change"] == 1.25
    assert review["items"][0]["description"] == "NEW"
    assert review["items"][0]["product_description"] == "NEW"


def test_explicit_null_name_does_not_fall_through_to_stale_aliases() -> None:
    item = {
        "name": None,
        "product_description": "OLD",
        "normalized_name": "OLD NORMALIZED",
        "description": "OLD",
        "final_price": 1.0,
    }

    assert item_description(item) == ""

    review = to_review_document({"receipt_metadata": {}, "items": [item]})
    assert review["items"][0]["description"] == ""
    assert review["items"][0]["product_description"] == ""


def test_next_schema_change_uses_legacy_alias_only_when_canonical_key_is_absent() -> None:
    receipt = _next_receipt()
    receipt["payment"].pop("change_returned")

    assert receipt_change(receipt) == 7.5


def test_legacy_receipt_keeps_compatibility_fallbacks() -> None:
    receipt = {
        "date": "2026-08-21",
        "currency": "EUR",
        "totals": {
            "subtotal": 10.0,
            "tax_total": 2.0,
            "grand_total": 12.0,
            "change": 1.5,
        },
        "payments": [{"method": "cash", "amount": 12.0}],
        "items": [{"description": "Legacy item", "line_total": 12.0}],
    }

    assert receipt_date(receipt) == "2026-08-21"
    assert receipt_currency(receipt) == "EUR"
    assert receipt_subtotal(receipt) == 10.0
    assert receipt_tax_total(receipt) == 2.0
    assert receipt_grand_total(receipt) == 12.0
    assert receipt_paid_total(receipt) == 12.0
    assert receipt_change(receipt) == 1.5
    assert receipt_payment_method(receipt) == "cash"
    assert item_description(receipt["items"][0]) == "Legacy item"
    assert item_line_total(receipt["items"][0]) == 12.0
