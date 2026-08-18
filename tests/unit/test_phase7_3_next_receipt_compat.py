from __future__ import annotations

import json
from pathlib import Path

from receipt_intelligence.receipt_compat import (
    to_legacy_validation_document,
    to_review_document,
    validation_issues,
)
from receipt_intelligence.services.review_service import apply_human_review
from receipt_intelligence.storage.fingerprints import receipt_core
from receipt_intelligence.storage.receipt_db import ReceiptDatabase
from receipt_intelligence.storage.repositories.review import ReviewRepository


def _next_receipt() -> dict:
    return {
        "merchant": {
            "name": "Modepark Röther",
            "address": {
                "street": "Josef-Landes-Straße 44",
                "postal_code": "87600",
                "city": "Kaufbeuren",
                "country": None,
            },
        },
        "receipt_metadata": {
            "date": "2017-12-17",
            "time": "19:55:55",
            "receipt_number": "8620",
            "currency": "EUR",
        },
        "items": [
            {
                "name": "KRAWATTE",
                "quantity": 1,
                "original_price": 14.99,
                "discount_amount": 0.75,
                "final_price": 14.24,
                "category_key": "clothing_shoes",
                "category_group": "Clothing",
            },
            {
                "name": "Louie Winter",
                "quantity": 1,
                "original_price": 120.0,
                "discount_amount": 24.0,
                "final_price": 96.0,
                "category_key": "clothing_shoes",
                "category_group": "Clothing",
            },
        ],
        "totals": {
            "final_purchase_total": {
                "final_purchase_total": 110.24,
                "currency": "EUR",
            },
            "net_amount": {"net_amount": 92.64, "currency": "EUR"},
        },
        "tax": {
            "vat_amount": {"vat_amount": 17.6, "currency": None},
            "vat_lines": [],
        },
        "payment": {
            "payment_method": "Lastschrift",
            "payment_received": {"payment_received": 110.24, "currency": "EUR"},
            "change_returned": None,
        },
        "validation": {
            "status": "review_required",
            "metrics": {"item_sum": 110.24, "final_purchase_total": 110.24},
            "checks": [
                {
                    "code": "ITEM_DISCOUNT_ARITHMETIC",
                    "status": "failed",
                    "severity": "review",
                    "message": "Item discount arithmetic is inconsistent.",
                    "details": [{"item_index": 0}],
                }
            ],
        },
    }


def test_review_projection_exposes_next_numbers_without_mutating_receipt() -> None:
    receipt = _next_receipt()
    projected = to_review_document(receipt)

    assert projected["merchant"]["address"] == ("Josef-Landes-Straße 44, 87600 Kaufbeuren")
    assert projected["currency"] == "EUR"
    assert projected["receipt_number"] == "8620"
    assert projected["totals"] == {
        **receipt["totals"],
        "subtotal": 92.64,
        "tax_total": 17.6,
        "grand_total": 110.24,
        "paid_total": 110.24,
        "change": None,
    }
    assert [item["line_total"] for item in projected["items"]] == [14.24, 96.0]
    assert [item["description"] for item in projected["items"]] == [
        "KRAWATTE",
        "Louie Winter",
    ]
    assert projected["validation"]["issues"][0]["code"] == ("ITEM_DISCOUNT_ARITHMETIC")
    assert isinstance(receipt["merchant"]["address"], dict)
    assert "line_total" not in receipt["items"][0]


def test_review_edits_update_canonical_next_paths_only() -> None:
    receipt, changed = apply_human_review(
        _next_receipt(),
        {
            "merchant_address": "New Street 1, 40225 Düsseldorf",
            "subtotal": "90.00",
            "tax_total": "20.24",
            "grand_total": "110.24",
            "paid_total": "110.24",
            "currency": "EUR",
        },
        [{"index": 0, "description": "KRAWATTE EDITED", "line_total": "14.24"}],
        {"status": "needs_review", "reviewer": "FT"},
    )

    assert receipt["merchant"]["address"] == {"formatted": "New Street 1, 40225 Düsseldorf"}
    assert receipt["totals"]["net_amount"]["net_amount"] == 90.0
    assert receipt["tax"]["vat_amount"]["vat_amount"] == 20.24
    assert receipt["totals"]["final_purchase_total"]["final_purchase_total"] == 110.24
    assert receipt["payment"]["payment_received"]["payment_received"] == 110.24
    assert receipt["items"][0]["name"] == "KRAWATTE EDITED"
    assert receipt["items"][0]["final_price"] == 14.24
    assert "line_total" not in receipt["items"][0]
    assert "items[0].final_price" in changed


def test_review_line_total_ignores_stale_projection_alias() -> None:
    receipt = _next_receipt()
    # Simulate a next-schema document that acquired the UI compatibility alias
    # during an earlier review/database round trip.
    receipt["items"][0]["line_total"] = 99.99

    projected = to_review_document(receipt)
    assert projected["items"][0]["line_total"] == 14.24

    updated, changed = apply_human_review(
        receipt,
        {},
        [{"index": 0, "line_total": "13.75"}],
        {"status": "needs_review", "reviewer": "FT"},
    )

    assert updated["items"][0]["final_price"] == 13.75
    assert "line_total" not in updated["items"][0]
    assert to_review_document(updated)["items"][0]["line_total"] == 13.75
    assert "items[0].final_price" in changed


def test_sql_import_prefers_next_final_price_over_stale_line_total(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "stale-line-total.sqlite3")
    receipt = _next_receipt()
    receipt["items"][0]["line_total"] = 99.99
    receipt["human_review"] = {"status": "approved", "reviewer": "FT"}

    imported = database.import_receipt(job_id="stale-line-total", receipt=receipt)
    with database.connect() as connection:
        stored = connection.execute(
            "SELECT line_total FROM receipt_items WHERE receipt_id = ? AND item_index = 0",
            (imported.receipt_db_id,),
        ).fetchone()

    assert stored["line_total"] == 14.24


def test_queue_projection_derives_summary_and_failed_checks_from_raw_json() -> None:
    row = ReviewRepository._present_queue_row(
        {
            "raw_json": json.dumps(_next_receipt()),
            "duplicate_candidates_json": "[]",
            "review_reason_codes_json": "[]",
            "grand_total": None,
            "item_count": 0,
            "issue_count": 0,
        }
    )

    assert row["grand_total"] == 110.24
    assert row["item_count"] == 2
    assert row["issue_count"] == 1
    assert row["reason_codes"] == ["ITEM_DISCOUNT_ARITHMETIC"]
    assert row["receipt"]["items"][0]["final_price"] == 14.24


def test_fingerprint_and_legacy_validator_projection_use_next_values() -> None:
    receipt = _next_receipt()
    core = receipt_core(receipt)
    validation_doc = to_legacy_validation_document(receipt)

    assert core["grand_total"] == 110.24
    assert core["item_count"] == 2
    assert "krawatte:14.24" in core["item_signature"]
    assert validation_doc["totals"]["grand_total"] == 110.24
    assert validation_doc["totals"]["tax_total"] == 17.6
    assert validation_doc["items"][1]["line_total"] == 96.0
    assert validation_issues(receipt["validation"])[0]["severity"] == "review"


def test_sql_import_projects_next_receipt_into_relational_columns(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipts.sqlite3")
    receipt = _next_receipt()
    receipt["human_review"] = {"status": "approved", "reviewer": "FT"}

    imported = database.import_receipt(job_id="modepark-next", receipt=receipt)
    with database.connect() as connection:
        stored = connection.execute(
            "SELECT receipt_date, receipt_time, currency, subtotal, tax_total, "
            "grand_total, paid_total, payment_method FROM receipts WHERE id = ?",
            (imported.receipt_db_id,),
        ).fetchone()
        items = connection.execute(
            "SELECT raw_name, line_total FROM receipt_items "
            "WHERE receipt_id = ? ORDER BY item_index",
            (imported.receipt_db_id,),
        ).fetchall()

    assert dict(stored) == {
        "receipt_date": "2017-12-17",
        "receipt_time": "19:55:55",
        "currency": "EUR",
        "subtotal": 92.64,
        "tax_total": 17.6,
        "grand_total": 110.24,
        "paid_total": 110.24,
        "payment_method": "Lastschrift",
    }
    assert [(row["raw_name"], row["line_total"]) for row in items] == [
        ("KRAWATTE", 14.24),
        ("Louie Winter", 96.0),
    ]
