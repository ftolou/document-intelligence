"""Migration tests for the rebuildable receipt-item embedding index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def _insert_receipt_and_item(database: ReceiptDatabase) -> tuple[int, int]:
    with database.connect() as connection:
        receipt_cursor = connection.execute(
            """
            INSERT INTO receipts(
                job_id,
                raw_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                "rag-migration-job",
                "{}",
                "2026-07-15T00:00:00+00:00",
                "2026-07-15T00:00:00+00:00",
            ),
        )
        receipt_id = int(receipt_cursor.lastrowid)

        item_cursor = connection.execute(
            """
            INSERT INTO receipt_items(
                receipt_id,
                item_index,
                raw_name,
                parser_item_type,
                embedding_text,
                raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                0,
                "DAMEN SNEAKER",
                "item",
                "damen sneaker",
                "{}",
            ),
        )
        item_id = int(item_cursor.lastrowid)
        connection.commit()

    return receipt_id, item_id


def test_rag_embedding_tables_have_expected_columns_and_indexes(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "rag-schema.db")

    with database.connect() as connection:
        embedding_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(rag_item_embeddings)").fetchall()
        }
        state_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(rag_index_state)").fetchall()
        }
        indexes = {
            row["name"]
            for row in connection.execute("PRAGMA index_list(rag_item_embeddings)").fetchall()
        }

    assert {
        "item_id",
        "embedding_model",
        "embedding_dimension",
        "document_text",
        "content_hash",
        "embedding",
        "updated_at",
    } == embedding_columns
    assert {
        "index_name",
        "embedding_model",
        "embedding_dimension",
        "indexed_count",
        "failed_count",
        "last_indexed_item_id",
        "last_completed_at",
        "last_error",
    } == state_columns
    assert {
        "idx_rag_item_embeddings_model",
        "idx_rag_item_embeddings_content_hash",
    } <= indexes


def test_embedding_rows_are_keyed_by_item_and_model_and_cascade_on_delete(
    tmp_path: Path,
) -> None:
    database = ReceiptDatabase(tmp_path / "rag-cascade.db")
    receipt_id, item_id = _insert_receipt_and_item(database)

    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO rag_item_embeddings(
                item_id,
                embedding_model,
                embedding_dimension,
                document_text,
                content_hash,
                embedding,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                "embeddinggemma",
                3,
                "Product description: DAMEN SNEAKER",
                "a" * 64,
                b"vector-bytes",
                "2026-07-15T00:00:00+00:00",
            ),
        )
        connection.commit()

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            connection.execute(
                """
                INSERT INTO rag_item_embeddings(
                    item_id,
                    embedding_model,
                    embedding_dimension,
                    document_text,
                    content_hash,
                    embedding,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    "embeddinggemma",
                    3,
                    "duplicate",
                    "b" * 64,
                    b"duplicate",
                    "2026-07-15T00:00:00+00:00",
                ),
            )

        connection.rollback()
        connection.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))
        connection.commit()

        remaining = connection.execute(
            "SELECT COUNT(*) AS count FROM rag_item_embeddings WHERE item_id = ?",
            (item_id,),
        ).fetchone()["count"]

    assert remaining == 0


def test_rag_index_state_accepts_progress_and_failure_metadata(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "rag-state.db")

    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO rag_index_state(
                index_name,
                embedding_model,
                embedding_dimension,
                indexed_count,
                failed_count,
                last_indexed_item_id,
                last_completed_at,
                last_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "approved_purchase_items",
                "embeddinggemma",
                768,
                125,
                2,
                140,
                "2026-07-15T00:00:00+00:00",
                "Two rows could not be embedded.",
            ),
        )
        connection.commit()

        row = connection.execute(
            "SELECT * FROM rag_index_state WHERE index_name = ?",
            ("approved_purchase_items",),
        ).fetchone()

    assert row["embedding_model"] == "embeddinggemma"
    assert row["embedding_dimension"] == 768
    assert row["indexed_count"] == 125
    assert row["failed_count"] == 2
    assert row["last_indexed_item_id"] == 140
