"""Storage migration tests for fresh and pre-Phase-3 SQLite databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from receipt_intelligence.storage.migrations import LATEST_SCHEMA_VERSION
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def test_fresh_database_applies_all_versioned_migrations(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "fresh.db")

    with database.connect() as connection:
        versions = [
            int(row["version"])
            for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]
        schema_version = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()["value"]
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }

    assert versions == list(range(1, LATEST_SCHEMA_VERSION + 1))
    assert schema_version == str(LATEST_SCHEMA_VERSION)
    assert {
        "receipts",
        "receipt_items",
        "review_queue",
        "schema_migrations",
        "rag_item_embeddings",
        "rag_index_state",
        "analytics_receipts",
        "analytics_purchase_items",
        "model_calls",
        "model_pricing",
        "receipt_review_history",
    } <= tables
    assert database.migrations.current_version() == LATEST_SCHEMA_VERSION


def test_legacy_database_is_adopted_without_losing_receipts(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta(key, value) VALUES ('schema_version', '2');

        CREATE TABLE receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE,
            merchant_name TEXT,
            merchant_normalized TEXT,
            receipt_date TEXT,
            receipt_time TEXT,
            currency TEXT,
            subtotal REAL,
            tax_total REAL,
            grand_total REAL,
            paid_total REAL,
            payment_method TEXT,
            review_status TEXT,
            reviewer TEXT,
            image_path TEXT,
            approved_receipt_path TEXT,
            source_receipt_path TEXT,
            raw_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE receipt_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
            item_index INTEGER NOT NULL,
            raw_name TEXT,
            normalized_name TEXT,
            category TEXT,
            category_group TEXT,
            category_key TEXT,
            quantity REAL,
            unit TEXT,
            unit_price REAL,
            line_total REAL,
            vat_rate TEXT,
            confidence REAL,
            review_status TEXT,
            embedding_text TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );

        INSERT INTO receipts(
            job_id, merchant_name, merchant_normalized, receipt_date, currency,
            grand_total, raw_json, created_at, updated_at
        ) VALUES (
            'legacy-job', 'REWE', 'rewe', '2026-06-18', 'EUR',
            20.0, '{}', '2026-06-18T10:00:00+00:00', '2026-06-18T10:00:00+00:00'
        );
        """
    )
    connection.commit()
    connection.close()

    database = ReceiptDatabase(database_path)

    with database.connect() as migrated:
        receipt = migrated.execute(
            "SELECT job_id, grand_total FROM receipts WHERE job_id='legacy-job'"
        ).fetchone()
        receipt_columns = {row["name"] for row in migrated.execute("PRAGMA table_info(receipts)")}
        item_columns = {row["name"] for row in migrated.execute("PRAGMA table_info(receipt_items)")}

    assert receipt["job_id"] == "legacy-job"
    assert receipt["grand_total"] == 20.0
    assert {
        "file_sha256",
        "content_fingerprint",
        "duplicate_status",
        "duplicate_of_receipt_id",
        "duplicate_score",
    } <= receipt_columns
    assert {"parser_item_type", "original_price", "discount_amount", "tax_code"} <= item_columns
    assert database.receipt_count() == 1


def test_migration_6_backfills_reviewed_semantics_from_item_json(tmp_path: Path) -> None:
    database_path = tmp_path / "semantic-backfill.db"
    database = ReceiptDatabase(database_path)
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO receipts(
                job_id, merchant_name, currency, raw_json, created_at, updated_at
            ) VALUES ('semantic-job', 'REWE', 'EUR', '{}', 'now', 'now')
            """
        )
        receipt_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
        connection.execute(
            """
            INSERT INTO receipt_items(
                receipt_id, item_index, raw_name, parser_item_type,
                embedding_text, raw_json, category_reason, semantic_description
            ) VALUES (?, 0, 'VITTEL', 'item', 'VITTEL', ?, NULL, NULL)
            """,
            (
                receipt_id,
                '{"category_reason":"Reviewed water category",'
                '"semantic_description":"Vittel is bottled mineral water."}',
            ),
        )
        connection.execute("DELETE FROM schema_migrations WHERE version = 6")
        connection.execute("UPDATE schema_meta SET value='5' WHERE key='schema_version'")
        connection.commit()

    database.migrations.migrate()
    with database.connect() as connection:
        row = connection.execute(
            "SELECT category_reason, semantic_description FROM receipt_items"
        ).fetchone()

    assert row["category_reason"] == "Reviewed water category"
    assert row["semantic_description"] == "Vittel is bottled mineral water."
