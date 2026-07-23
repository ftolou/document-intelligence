from __future__ import annotations

from pathlib import Path

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
    assert version == "6"


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
