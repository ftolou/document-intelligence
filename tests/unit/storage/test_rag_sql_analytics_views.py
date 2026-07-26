from __future__ import annotations

from pathlib import Path

from receipt_intelligence.storage.migrations import LATEST_SCHEMA_VERSION
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def test_rag_sql_analytics_views_exist_on_fresh_database(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "views.db")
    with database.connect() as connection:
        views = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='view'"
            ).fetchall()
        }
        version = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()["value"]

    assert {"analytics_receipts", "analytics_purchase_items"} <= views
    assert version == str(LATEST_SCHEMA_VERSION)


def test_purchase_item_view_exposes_reviewed_product_semantics(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "semantic-view.db")
    with database.connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(analytics_purchase_items)").fetchall()
        }
        item_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(receipt_items)").fetchall()
        }

    assert {"category_reason", "semantic_description"} <= columns
    assert {"category_reason", "semantic_description"} <= item_columns


def test_rejected_receipt_is_excluded_even_when_approved_path_exists(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "rejected.db")
    approved_path = tmp_path / "approved_receipt.json"
    approved_path.write_text("{}", encoding="utf-8")
    database.import_receipt(
        job_id="job-rejected",
        approved_receipt_path=approved_path,
        receipt={
            "merchant": {"name": "Aral Tankstelle"},
            "date": "2020-03-05",
            "currency": "EUR",
            "totals": {"grand_total": 6.0},
            "human_review": {"status": "rejected"},
            "items": [
                {
                    "product_description": "HEETS TURQUOISE",
                    "category": "item",
                    "line_total": 6.0,
                }
            ],
        },
    )

    with database.connect() as connection:
        receipt_count = connection.execute(
            "SELECT COUNT(*) AS n FROM analytics_receipts"
        ).fetchone()["n"]
        item_count = connection.execute(
            "SELECT COUNT(*) AS n FROM analytics_purchase_items"
        ).fetchone()["n"]

    assert receipt_count == 0
    assert item_count == 0


def test_approved_receipt_excludes_rejected_item_rows(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "item-boundary.db")
    database.import_receipt(
        job_id="job-approved",
        receipt={
            "merchant": {"name": "REWE"},
            "date": "2026-07-25",
            "currency": "EUR",
            "totals": {"grand_total": 5.0},
            "human_review": {"status": "approved"},
            "items": [
                {
                    "product_description": "VITTEL",
                    "category": "item",
                    "review_status": "approved",
                    "line_total": 5.0,
                },
                {
                    "product_description": "None",
                    "category": "item",
                    "review_status": "rejected",
                    "line_total": 0.0,
                },
            ],
        },
    )

    with database.connect() as connection:
        receipt = connection.execute("SELECT item_count FROM analytics_receipts").fetchone()
        descriptions = [
            row["description"]
            for row in connection.execute(
                "SELECT description FROM analytics_purchase_items ORDER BY item_id"
            ).fetchall()
        ]

    assert receipt["item_count"] == 1
    assert descriptions == ["VITTEL"]
