"""SQLite implementation of semantic candidate retrieval."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from receipt_intelligence.rag.ports import SemanticSearchCandidate
from receipt_intelligence.rag.vector_codec import blob_to_vector
from receipt_intelligence.storage.connection import SQLiteConnectionFactory

_PURCHASE_ITEM_TYPES = ("item", "product", "purchase_item", "purchased_product")
_APPROVED_RECEIPT_PREDICATE = (
    "lower(COALESCE(r.review_status, '')) IN ('approved', 'accepted', 'complete', 'completed')"
)


class SQLiteSemanticSearchRepository:
    """Load semantic and FTS candidate rows without leaking SQLite to RAG."""

    def __init__(self, database_path: Path | str) -> None:
        self.connections = SQLiteConnectionFactory(database_path)

    def load_candidates(
        self,
        *,
        embedding_model: str,
        approved_only: bool,
        merchant: str | None,
        category: str | None,
        item_ids: Sequence[int] | None,
    ) -> list[SemanticSearchCandidate]:
        clauses, parameters = self._structured_filters(
            approved_only=approved_only,
            merchant=merchant,
            category=category,
            item_ids=item_ids,
            embedding_model=embedding_model,
        )
        sql = f"""
            SELECT
                e.item_id,
                e.embedding_dimension,
                e.embedding,
                i.receipt_id,
                i.raw_name AS description,
                i.normalized_name AS normalized_description,
                COALESCE(i.category, i.category_key, i.category_group) AS category,
                i.raw_json AS item_raw_json,
                i.parser_item_type,
                i.line_total,
                i.unit_price,
                r.merchant_name AS merchant,
                r.receipt_date,
                r.currency
            FROM rag_item_embeddings AS e
            JOIN receipt_items AS i ON i.id = e.item_id
            JOIN receipts AS r ON r.id = i.receipt_id
            WHERE {" AND ".join(clauses)}
            ORDER BY e.item_id
        """
        with self.connections.connect_read_only() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def load_fts_ranks(
        self,
        *,
        fts_query: str,
        approved_only: bool,
        merchant: str | None,
        category: str | None,
        item_ids: Sequence[int] | None,
        maximum_results: int,
    ) -> dict[int, int]:
        if not fts_query:
            return {}
        with self.connections.connect_read_only() as connection:
            available = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='receipt_item_fts'"
            ).fetchone()
            if not available:
                return {}
            clauses, parameters = self._structured_filters(
                approved_only=approved_only,
                merchant=merchant,
                category=category,
                item_ids=item_ids,
                embedding_model=None,
            )
            sql = f"""
                SELECT receipt_item_fts.item_id,
                       bm25(receipt_item_fts, 0.0, 0.0, 0.0, 0.0, 10.0, 12.0, 0.0, 0.0)
                           AS lexical_bm25
                FROM receipt_item_fts
                JOIN receipt_items AS i ON i.id = receipt_item_fts.item_id
                JOIN receipts AS r ON r.id = i.receipt_id
                WHERE receipt_item_fts MATCH ?
                  AND {" AND ".join(clauses)}
                ORDER BY lexical_bm25 ASC, receipt_item_fts.item_id ASC
                LIMIT ?
            """
            try:
                rows = connection.execute(
                    sql,
                    [fts_query, *parameters, int(maximum_results)],
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
        return {int(row["item_id"]): rank for rank, row in enumerate(rows, start=1)}

    @staticmethod
    def _candidate_from_row(row: sqlite3.Row) -> SemanticSearchCandidate:
        dimension = int(row["embedding_dimension"])
        try:
            vector = tuple(blob_to_vector(bytes(row["embedding"]), dimension=dimension))
        except (TypeError, ValueError):
            vector = None
        return SemanticSearchCandidate(
            item_id=int(row["item_id"]),
            embedding_dimension=dimension,
            vector=vector,
            receipt_id=int(row["receipt_id"]),
            description=row["description"],
            normalized_description=row["normalized_description"],
            category=row["category"],
            item_raw_json=row["item_raw_json"],
            parser_item_type=row["parser_item_type"],
            line_total=float(row["line_total"]) if row["line_total"] is not None else None,
            unit_price=float(row["unit_price"]) if row["unit_price"] is not None else None,
            merchant=row["merchant"],
            receipt_date=row["receipt_date"],
            currency=row["currency"],
        )

    @staticmethod
    def _structured_filters(
        *,
        approved_only: bool,
        merchant: str | None,
        category: str | None,
        item_ids: Sequence[int] | None,
        embedding_model: str | None,
    ) -> tuple[list[str], list[object]]:
        clauses: list[str] = []
        parameters: list[object] = []
        if embedding_model is not None:
            clauses.append("e.embedding_model = ?")
            parameters.append(embedding_model)
        clauses.append("lower(COALESCE(i.parser_item_type, 'item')) IN (?, ?, ?, ?)")
        parameters.extend(_PURCHASE_ITEM_TYPES)
        clauses.append("lower(COALESCE(i.review_status, '')) NOT IN ('rejected', 'needs_review')")
        if approved_only:
            clauses.append(_APPROVED_RECEIPT_PREDICATE)

        normalized_merchant = " ".join(str(merchant or "").split()).strip()
        if normalized_merchant:
            clauses.append(
                "(lower(COALESCE(r.merchant_normalized, '')) = lower(?) OR "
                "lower(COALESCE(r.merchant_name, '')) = lower(?))"
            )
            parameters.extend([normalized_merchant, normalized_merchant])

        normalized_category = " ".join(str(category or "").split()).strip()
        if normalized_category:
            clauses.append(
                "lower(COALESCE(i.category_key, i.category_group, i.category, '')) = lower(?)"
            )
            parameters.append(normalized_category)

        selected = [int(item_id) for item_id in item_ids or ()]
        if selected:
            placeholders = ", ".join("?" for _ in selected)
            clauses.append(f"i.id IN ({placeholders})")
            parameters.extend(selected)
        return clauses, parameters


__all__ = ["SQLiteSemanticSearchRepository"]
