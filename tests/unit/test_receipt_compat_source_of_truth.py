from __future__ import annotations

from receipt_intelligence.receipt_compat import (
    item_line_total,
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
        },
        "tax": {
            "vat_amount": {"vat_amount": None, "currency": "EUR"},
        },
        "payment": {
            "payment_method": None,
            "payment_received": {"payment_received": None, "currency": "EUR"},
        },
        "payments": [{"method": "cash", "amount": 99.0}],
        "items": [
            {
                "name": "Canonical item",
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
    assert receipt_payment_method(receipt) is None
    assert item_line_total(receipt["items"][0]) is None

    review = to_review_document(receipt)
    assert review["totals"]["subtotal"] is None
    assert review["totals"]["tax_total"] is None
    assert review["totals"]["grand_total"] == 12.0
    assert review["totals"]["paid_total"] is None
    assert review["payment_method"] is None
    assert review["items"][0]["line_total"] is None


def test_legacy_receipt_keeps_compatibility_fallbacks() -> None:
    receipt = {
        "date": "2026-08-21",
        "currency": "EUR",
        "totals": {
            "subtotal": 10.0,
            "tax_total": 2.0,
            "grand_total": 12.0,
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
    assert receipt_payment_method(receipt) == "cash"
    assert item_line_total(receipt["items"][0]) == 12.0
