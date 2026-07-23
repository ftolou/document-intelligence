"""Incremental semantic indexer for approved purchase-item rows."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from receipt_intelligence.rag.item_documents import (
    UnindexableItemDescriptionError,
    build_item_embedding_document,
)
from receipt_intelligence.rag.models import (
    EmbeddingBatchResult,
    ItemEmbeddingDocument,
    ItemEmbeddingIndexReport,
)
from receipt_intelligence.rag.vector_codec import vector_to_blob
from receipt_intelligence.storage.connection import SQLiteConnectionFactory
from receipt_intelligence.storage.migrations import MigrationRunner

_PURCHASE_ITEM_TYPES = ("item", "product", "purchase_item", "purchased_product")
_DEFAULT_INDEX_NAME = "approved_purchase_items"


class EmbeddingClient(Protocol):
    model: str

    def embed(self, texts: list[str]) -> EmbeddingBatchResult: ...


class ItemEmbeddingIndexer:
    """Build and update the derived item-embedding index.

    The source receipt database remains authoritative. Embeddings are rebuilt
    only when their canonical semantic document hash changes or when a different
    embedding model is selected.
    """

    def __init__(
        self,
        *,
        database_path: Path | str,
        embedding_client: EmbeddingClient,
        batch_size: int = 32,
        index_name: str = _DEFAULT_INDEX_NAME,
        approved_only: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        normalized_name = str(index_name or "").strip()
        if not normalized_name:
            raise ValueError("index_name must not be empty.")

        self.database_path = Path(database_path)
        self.embedding_client = embedding_client
        self.batch_size = int(batch_size)
        self.index_name = normalized_name
        self.approved_only = bool(approved_only)
        self.connections = SQLiteConnectionFactory(self.database_path)
        self.migrations = MigrationRunner(self.connections)

    def rebuild(self, *, force: bool = False) -> ItemEmbeddingIndexReport:
        """Incrementally index all eligible rows.

        ``force=True`` re-embeds every eligible item for the active model.
        Otherwise unchanged ``content_hash`` values are skipped.
        """

        self.migrations.migrate()
        documents = self._load_documents()
        pruned = self._prune_embeddings(
            eligible_item_ids={document.item_id for document in documents}
        )
        existing = self._existing_hashes()

        pending = [
            document
            for document in documents
            if force or existing.get(document.item_id) != document.content_hash
        ]
        unchanged = len(documents) - len(pending)

        embedded = 0
        failed = 0
        batches = 0
        dimension = self._known_dimension()
        last_indexed_item_id: int | None = None
        errors: list[str] = []

        if not pending:
            self._write_state(
                dimension=dimension,
                indexed_count=0,
                failed_count=0,
                last_indexed_item_id=None,
                last_error=None,
            )
            return ItemEmbeddingIndexReport(
                index_name=self.index_name,
                model=self.embedding_client.model,
                eligible_items=len(documents),
                embedded=0,
                unchanged=unchanged,
                failed=0,
                pruned=pruned,
                dimension=dimension,
                batches=0,
            )

        for batch in self._batches(pending):
            batches += 1
            try:
                result = self.embedding_client.embed([document.text for document in batch])
                dimension = self._validate_dimension(result, expected=dimension)
                self._store_batch(batch, result)
                embedded += len(batch)
                last_indexed_item_id = batch[-1].item_id
                self._write_state(
                    dimension=dimension,
                    indexed_count=embedded,
                    failed_count=failed,
                    last_indexed_item_id=last_indexed_item_id,
                    last_error=errors[-1] if errors else None,
                )
            except Exception as exc:  # isolate provider/storage failures by batch
                failed += len(batch)
                message = (
                    f"items {batch[0].item_id}-{batch[-1].item_id}: {type(exc).__name__}: {exc}"
                )
                errors.append(message)
                self._write_state(
                    dimension=dimension,
                    indexed_count=embedded,
                    failed_count=failed,
                    last_indexed_item_id=last_indexed_item_id,
                    last_error=message,
                )

        return ItemEmbeddingIndexReport(
            index_name=self.index_name,
            model=self.embedding_client.model,
            eligible_items=len(documents),
            embedded=embedded,
            unchanged=unchanged,
            failed=failed,
            pruned=pruned,
            dimension=dimension,
            batches=batches,
            last_indexed_item_id=last_indexed_item_id,
            errors=errors,
        )

    def index_item_ids(
        self,
        item_ids: Iterable[int],
        *,
        force: bool = False,
    ) -> ItemEmbeddingIndexReport:
        """Index a selected set of SQL item IDs, suitable for post-approval use."""

        selected = sorted({int(item_id) for item_id in item_ids if int(item_id) > 0})
        if not selected:
            return ItemEmbeddingIndexReport(
                index_name=self.index_name,
                model=self.embedding_client.model,
                eligible_items=0,
                embedded=0,
                unchanged=0,
                failed=0,
                batches=0,
            )

        self.migrations.migrate()
        documents = self._load_documents(item_ids=selected)
        pruned = self._prune_embeddings(
            eligible_item_ids={document.item_id for document in documents},
            scope_item_ids=set(selected),
        )
        existing = self._existing_hashes(item_ids=selected)
        pending = [
            document
            for document in documents
            if force or existing.get(document.item_id) != document.content_hash
        ]
        unchanged = len(documents) - len(pending)
        dimension = self._known_dimension()
        embedded = 0
        failed = 0
        batches = 0
        last_indexed_item_id: int | None = None
        errors: list[str] = []

        for batch in self._batches(pending):
            batches += 1
            try:
                result = self.embedding_client.embed([document.text for document in batch])
                dimension = self._validate_dimension(result, expected=dimension)
                self._store_batch(batch, result)
                embedded += len(batch)
                last_indexed_item_id = batch[-1].item_id
            except Exception as exc:
                failed += len(batch)
                errors.append(
                    f"items {batch[0].item_id}-{batch[-1].item_id}: {type(exc).__name__}: {exc}"
                )

        self._write_state(
            dimension=dimension,
            indexed_count=embedded,
            failed_count=failed,
            last_indexed_item_id=last_indexed_item_id,
            last_error=errors[-1] if errors else None,
        )
        return ItemEmbeddingIndexReport(
            index_name=self.index_name,
            model=self.embedding_client.model,
            eligible_items=len(documents),
            embedded=embedded,
            unchanged=unchanged,
            failed=failed,
            pruned=pruned,
            dimension=dimension,
            batches=batches,
            last_indexed_item_id=last_indexed_item_id,
            errors=errors,
        )

    def _load_documents(self, item_ids: list[int] | None = None) -> list[ItemEmbeddingDocument]:
        clauses = [
            "trim(COALESCE(i.raw_name, '')) <> ''",
            "lower(COALESCE(i.parser_item_type, 'item')) IN (?, ?, ?, ?)",
        ]
        parameters: list[object] = list(_PURCHASE_ITEM_TYPES)

        if self.approved_only:
            clauses.append(
                "(r.approved_receipt_path IS NOT NULL OR "
                "lower(COALESCE(r.review_status, '')) IN "
                "('approved', 'accepted', 'saved', 'complete', 'completed'))"
            )
        if item_ids:
            placeholders = ", ".join("?" for _ in item_ids)
            clauses.append(f"i.id IN ({placeholders})")
            parameters.extend(item_ids)

        sql = f"""
            SELECT
                i.id AS item_id,
                i.receipt_id,
                i.raw_name AS description,
                i.normalized_name AS description_normalized,
                i.category AS category,
                i.category_group AS category_group,
                i.category_key AS category_key,
                i.category_reason AS category_reason,
                i.semantic_description AS semantic_description,
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

        documents: list[ItemEmbeddingDocument] = []
        for row in rows:
            try:
                documents.append(build_item_embedding_document(dict(row)))
            except UnindexableItemDescriptionError:
                continue
        return documents

    def _prune_embeddings(
        self,
        *,
        eligible_item_ids: set[int],
        scope_item_ids: set[int] | None = None,
    ) -> int:
        """Delete stale embeddings for rows no longer eligible for this index."""

        with self.connections.connect() as connection:
            rows = connection.execute(
                "SELECT item_id FROM rag_item_embeddings WHERE embedding_model = ?",
                (self.embedding_client.model,),
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
                [self.embedding_client.model, *stale_ids],
            )
            connection.commit()
        return len(stale_ids)

    def _existing_hashes(self, item_ids: list[int] | None = None) -> dict[int, str]:
        sql = """
            SELECT item_id, content_hash
            FROM rag_item_embeddings
            WHERE embedding_model = ?
        """
        parameters: list[object] = [self.embedding_client.model]
        if item_ids:
            placeholders = ", ".join("?" for _ in item_ids)
            sql += f" AND item_id IN ({placeholders})"
            parameters.extend(item_ids)
        with self.connections.connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return {int(row["item_id"]): str(row["content_hash"]) for row in rows}

    def _known_dimension(self) -> int | None:
        with self.connections.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT embedding_dimension
                FROM rag_item_embeddings
                WHERE embedding_model = ?
                """,
                (self.embedding_client.model,),
            ).fetchall()
        dimensions = {int(row["embedding_dimension"]) for row in rows}
        if len(dimensions) > 1:
            raise ValueError(
                f"Stored embeddings for {self.embedding_client.model!r} have mixed dimensions."
            )
        return next(iter(dimensions), None)

    @staticmethod
    def _validate_dimension(result: EmbeddingBatchResult, *, expected: int | None) -> int:
        if result.count <= 0 or result.dimension <= 0:
            raise ValueError("Embedding provider returned an empty batch.")
        if expected is not None and result.dimension != expected:
            raise ValueError(f"Embedding dimension changed from {expected} to {result.dimension}.")
        return result.dimension

    def _store_batch(
        self,
        documents: list[ItemEmbeddingDocument],
        result: EmbeddingBatchResult,
    ) -> None:
        if result.count != len(documents):
            raise ValueError("Embedding result count does not match document batch.")
        now = datetime.now(UTC).isoformat()
        records = [
            (
                document.item_id,
                self.embedding_client.model,
                result.dimension,
                document.text,
                document.content_hash,
                sqlite3.Binary(vector_to_blob(vector)),
                now,
            )
            for document, vector in zip(documents, result.vectors, strict=True)
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
                records,
            )
            connection.commit()

    def _write_state(
        self,
        *,
        dimension: int | None,
        indexed_count: int,
        failed_count: int,
        last_indexed_item_id: int | None,
        last_error: str | None,
    ) -> None:
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
                    self.index_name,
                    self.embedding_client.model,
                    dimension,
                    indexed_count,
                    failed_count,
                    last_indexed_item_id,
                    datetime.now(UTC).isoformat(),
                    last_error,
                ),
            )
            connection.commit()

    def _batches(
        self,
        documents: list[ItemEmbeddingDocument],
    ) -> Iterable[list[ItemEmbeddingDocument]]:
        for offset in range(0, len(documents), self.batch_size):
            yield documents[offset : offset + self.batch_size]
