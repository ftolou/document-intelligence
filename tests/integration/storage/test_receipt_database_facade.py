"""Verify the compatibility facade delegates to the focused repositories."""

from __future__ import annotations

from pathlib import Path

from receipt_intelligence.storage.receipt_db import ReceiptDatabase
from receipt_intelligence.storage.repositories import (
    AnalyticsRepository,
    ReceiptRepository,
    ReviewRepository,
    SearchRepository,
)

RECEIPT = {
    "merchant": {"name": "dm-drogerie markt"},
    "date": "2026-06-12",
    "currency": "EUR",
    "totals": {"grand_total": 10.0, "paid_total": 10.0},
    "items": [
        {
            "description": "HEAD&SHOULDERS CLASSIC",
            "category": "item",
            "category_key": "personal_care",
            "line_total": 3.95,
        }
    ],
    "human_review": {"status": "approved"},
}


def test_facade_exposes_repositories_and_preserves_public_behavior(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipts.db")

    assert isinstance(database.receipts, ReceiptRepository)
    assert isinstance(database.analytics, AnalyticsRepository)
    assert isinstance(database.review, ReviewRepository)
    assert isinstance(database.search, SearchRepository)

    imported = database.import_receipt(job_id="dm-1", receipt=RECEIPT)
    matches = database.search_items(semantic_query="shampoo", limit=25)
    aggregate = database.aggregate_receipts(merchant="dm")

    assert imported.item_count == 1
    assert database.receipt_count() == 1
    assert database.item_count() == 1
    assert matches[0]["job_id"] == "dm-1"
    assert matches[0]["category"] == "personal_care/shampoo"
    assert aggregate["value"] == 10.0
    assert aggregate["currency"] == "EUR"
