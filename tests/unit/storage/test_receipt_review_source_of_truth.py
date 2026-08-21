from __future__ import annotations

from pathlib import Path

from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def _next_receipt_with_stale_aliases() -> dict:
    return {
        "merchant": {"name": "REWE", "provider_note": "keep-me"},
        "receipt_metadata": {
            "date": "2026-08-21",
            "currency": "EUR",
        },
        "date": "1999-01-01",
        "currency": "USD",
        "totals": {
            "net_amount": {"net_amount": 10.0, "currency": "EUR"},
            "subtotal": 99.0,
            "final_purchase_total": {
                "final_purchase_total": 12.0,
                "currency": "EUR",
            },
            "grand_total": 99.0,
            "tax_total": 99.0,
            "paid_total": 99.0,
        },
        "tax": {
            "vat_amount": {"vat_amount": 2.0, "currency": "EUR"},
        },
        "payment": {
            "payment_method": "card",
            "payment_received": {"payment_received": 12.0, "currency": "EUR"},
        },
        "payments": [{"method": "cash", "amount": 99.0}],
        "items": [
            {
                "name": "CANONICAL ITEM",
                "product_description": "STALE ITEM",
                "description": "STALE ITEM",
                "normalized_name": "stale normalized",
                "final_price": 4.25,
                "line_total": 99.0,
                "total": 99.0,
                "amount": 99.0,
                "parser_item_type": "item",
                "receipt_row_type": "discount",
                "line_type": "discount",
                "category_group": "Food",
                "category_key": "fruit",
                "product_category": "stale/category",
                "vat_rate": "7",
                "tax_rate": "19",
                "confidence": 0.4,
                "category_confidence": 0.9,
                "provider_note": "keep-item-metadata",
            }
        ],
        "human_review": {"status": "needs_review"},
    }


def test_relational_values_survive_get_save_round_trip_over_stale_json(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    imported = database.import_receipt(
        job_id="job-relational-authority",
        receipt=_next_receipt_with_stale_aliases(),
    )
    receipt_id = imported.receipt_db_id

    with database.connect() as connection:
        connection.execute(
            """
            UPDATE receipts
            SET currency=NULL,
                subtotal=10.0,
                tax_total=NULL,
                grand_total=12.0,
                paid_total=NULL,
                payment_method=NULL,
                review_status='needs_review'
            WHERE id=?
            """,
            (receipt_id,),
        )
        connection.execute(
            """
            UPDATE receipt_items
            SET raw_name='CANONICAL ITEM',
                normalized_name=NULL,
                category=NULL,
                parser_item_type=NULL,
                category_group=NULL,
                category_key=NULL,
                category_reason=NULL,
                semantic_description=NULL,
                line_total=NULL,
                vat_rate=NULL,
                confidence=NULL,
                review_status='needs_review'
            WHERE receipt_id=?
            """,
            (receipt_id,),
        )
        connection.commit()

    document = database.get_receipt_edit_document(receipt_id)
    assert document is not None
    assert document["merchant"]["name"] == "REWE"
    assert document["merchant"]["provider_note"] == "keep-me"
    assert document["receipt_metadata"]["currency"] is None
    assert document["currency"] is None
    assert document["totals"]["subtotal"] == 10.0
    assert document["totals"]["tax_total"] is None
    assert document["totals"]["grand_total"] == 12.0
    assert document["totals"]["paid_total"] is None
    assert document["payment"]["payment_method"] is None
    assert document["payments"][0]["method"] is None
    assert document["payments"][0]["amount"] is None

    item = document["items"][0]
    assert item["name"] == "CANONICAL ITEM"
    assert item["product_description"] == "CANONICAL ITEM"
    assert item["description"] == "CANONICAL ITEM"
    assert item["normalized_name"] is None
    assert item["final_price"] is None
    assert item["line_total"] is None
    assert item["total"] is None
    assert item["amount"] is None
    assert item["parser_item_type"] is None
    assert item["receipt_row_type"] is None
    assert item["line_type"] is None
    assert item["product_category"] is None
    assert item["category_path"] is None
    assert item["vat_rate"] is None
    assert item["tax_rate"] is None
    assert item["confidence"] is None
    assert item["category_confidence"] is None
    assert item["provider_note"] == "keep-item-metadata"

    identity = dict(document["_database"])
    result = database.update_receipt_from_review(
        receipt_id,
        document,
        expected_job_id=str(identity["job_id"]),
        expected_updated_at=str(identity["updated_at"]),
    )
    assert result["receipt_db_id"] == receipt_id

    with database.connect() as connection:
        receipt_row = connection.execute(
            """
            SELECT currency, subtotal, tax_total, grand_total, paid_total, payment_method
            FROM receipts WHERE id=?
            """,
            (receipt_id,),
        ).fetchone()
        item_row = connection.execute(
            """
            SELECT raw_name, normalized_name, category, parser_item_type,
                   category_group, category_key, category_reason, semantic_description,
                   line_total, vat_rate, confidence
            FROM receipt_items WHERE receipt_id=?
            """,
            (receipt_id,),
        ).fetchone()

    assert receipt_row["currency"] is None
    assert receipt_row["subtotal"] == 10.0
    assert receipt_row["tax_total"] is None
    assert receipt_row["grand_total"] == 12.0
    assert receipt_row["paid_total"] is None
    assert receipt_row["payment_method"] is None
    assert item_row["raw_name"] == "CANONICAL ITEM"
    assert item_row["normalized_name"] is None
    assert item_row["category"] is None
    assert item_row["parser_item_type"] is None
    assert item_row["category_group"] is None
    assert item_row["category_key"] is None
    assert item_row["category_reason"] is None
    assert item_row["semantic_description"] is None
    assert item_row["line_total"] is None
    assert item_row["vat_rate"] is None
    assert item_row["confidence"] is None
