"""SQLite implementation of the semantic-index persistence contract."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from receipt_intelligence.rag.ports import (
    IndexableItemSource,
    SemanticIndexState,
    StoredItemEmbedding,
)
from receipt_intelligence.rag.vector_codec import vector_to_blob
from receipt_intelligence.storage.connection import SQLiteConnectionFactory

_PURCHASE_ITEM_TYPES = ("item", "product", "purchase_item", "purchased_product")
_APPROVED_RECEIPT_PREDICATE = (
    "(r.approved_receipt_path IS NOT NULL OR "
    "lower(COALESCE(r.review_status, '')) IN "
    "('approved', 'accepted', 'saved', 'complete', 'completed'))"
)


class SQLiteSemanticIndexRepository:
    """Persist the rebuildable semantic item index in SQLite."""

    def __init__(self, database_path: Path | str) -> None:
        self.connections = SQLiteConnectionFactory(database_path)

    def load_indexable_items(
        self,
        *,
        approved_only: bool,
        item_ids: Sequence[int] | None = None,
    ) -> list[IndexableItemSource]:
        clauses = [
            "trim(COALESCE(i.raw_name, '')) <> ''",
            "lower(COALESCE(i.parser_item_type, 'item')) IN (?, ?, ?, ?)",
            "lower(COALESCE(i.review_status, '')) NOT IN ('rejected', 'needs_review')",
        ]
        parameters: list[object] = list(_PURCHASE_ITEM_TYPES)
        if approved_only:
            clauses.append(_APPROVED_RECEIPT_PREDICATE)
        selected = [int(item_id) for item_id in item_ids or ()]
        if selected:
            placeholders = ", ".join("?" for _ in selected)
            clauses.append(f"i.id IN ({placeholders})")
            parameters.extend(selected)

        sql = f"""
            SELECT
                i.id AS item_id,
                i.receipt_id,
                i.raw_name AS description,
                i.normalized_name AS description_normalized,
                i.category,
                i.category_group,
                i.category_key,
                i.category_reason,
                i.semantic_description,
                i.raw_json AS item_raw_json,
                r.merchant_name AS merchant,
                i.parser_item_type
            FROM receipt_items AS i
            JOIN receipts AS r ON r.id = i.receipt_id
            WHERE {" AND ".join(clauses)}
            ORDER BY i.id
        """
        with self.connections.connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [
            IndexableItemSource(
                item_id=int(row["item_id"]),
                receipt_id=int(row["receipt_id"]) if row["receipt_id"] is not None else None,
                description=row["description"],
                description_normalized=row["description_normalized"],
                category=row["category"],
                category_group=row["category_group"],
                category_key=row["category_key"],
                category_reason=row["category_reason"],
                semantic_description=row["semantic_description"],
                item_raw_json=row["item_raw_json"],
                merchant=row["merchant"],
                parser_item_type=row["parser_item_type"],
            )
            for row in rows
        ]

    def prune_embeddings(
        self,
        *,
        embedding_model: str,
        eligible_item_ids: set[int],
        scope_item_ids: set[int] | None = None,
    ) -> int:
        with self.connections.connect() as connection:
            rows = connection.execute(
                "SELECT item_id FROM rag_item_embeddings WHERE embedding_model = ?",
                (embedding_model,),
            ).fetchall()
            existing_ids = {int(row["item_id"]) for row in rows}
            if scope_item_ids is not None:
                existing_ids &= scope_item_ids
            stale_ids = sorted(existing_ids - eligible_item_ids)
            if not stale_ids:
                return 0
            placeholders = ", ".join("?" for _ in stale_ids)
            connection.execute(
                f"DELETE FROM rag_item_embeddings "
                f"WHERE embedding_model = ? AND item_id IN ({placeholders})",
                [embedding_model, *stale_ids],
            )
            connection.commit()
        return len(stale_ids)

    def existing_hashes(
        self,
        *,
        embedding_model: str,
        item_ids: Sequence[int] | None = None,
    ) -> dict[int, str]:
        sql = """
            SELECT item_id, content_hash
            FROM rag_item_embeddings
            WHERE embedding_model = ?
        """
        parameters: list[object] = [embedding_model]
        selected = [int(item_id) for item_id in item_ids or ()]
        if selected:
            placeholders = ", ".join("?" for _ in selected)
            sql += f" AND item_id IN ({placeholders})"
            parameters.extend(selected)
        with self.connections.connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return {int(row["item_id"]): str(row["content_hash"]) for row in rows}

    def known_dimension(self, *, embedding_model: str) -> int | None:
        with self.connections.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT embedding_dimension
                FROM rag_item_embeddings
                WHERE embedding_model = ?
                """,
                (embedding_model,),
            ).fetchall()
        dimensions = {int(row["embedding_dimension"]) for row in rows}
        if len(dimensions) > 1:
            raise ValueError(f"Stored embeddings for {embedding_model!r} have mixed dimensions.")
        return next(iter(dimensions), None)

    def store_embeddings(self, records: Sequence[StoredItemEmbedding]) -> None:
        if not records:
            return
        values = [
            (
                record.item_id,
                record.embedding_model,
                record.embedding_dimension,
                record.document_text,
                record.content_hash,
                sqlite3.Binary(vector_to_blob(record.vector)),
                record.updated_at,
            )
            for record in records
        ]
        with self.connections.connect() as connection:
            connection.executemany(
                """
                INSERT INTO rag_item_embeddings(
                    item_id, embedding_model, embedding_dimension,
                    document_text, content_hash, embedding, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id, embedding_model) DO UPDATE SET
                    embedding_dimension = excluded.embedding_dimension,
                    document_text = excluded.document_text,
                    content_hash = excluded.content_hash,
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at
                """,
                values,
            )
            connection.commit()

    def save_state(self, state: SemanticIndexState) -> None:
        with self.connections.connect() as connection:
            connection.execute(
                """
                INSERT INTO rag_index_state(
                    index_name, embedding_model, embedding_dimension,
                    indexed_count, failed_count, last_indexed_item_id,
                    last_completed_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(index_name) DO UPDATE SET
                    embedding_model = excluded.embedding_model,
                    embedding_dimension = excluded.embedding_dimension,
                    indexed_count = excluded.indexed_count,
                    failed_count = excluded.failed_count,
                    last_indexed_item_id = excluded.last_indexed_item_id,
                    last_completed_at = excluded.last_completed_at,
                    last_error = excluded.last_error
                """,
                (
                    state.index_name,
                    state.embedding_model,
                    state.embedding_dimension,
                    state.indexed_count,
                    state.failed_count,
                    state.last_indexed_item_id,
                    state.last_completed_at,
                    state.last_error,
                ),
            )
            connection.commit()


__all__ = ["SQLiteSemanticIndexRepository"]
