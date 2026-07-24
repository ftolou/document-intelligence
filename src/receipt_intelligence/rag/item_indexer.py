"""Incremental semantic indexer for approved purchase-item rows."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
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
from receipt_intelligence.rag.ports import (
    SemanticIndexRepository,
    SemanticIndexState,
    StoredItemEmbedding,
)

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
        repository: SemanticIndexRepository,
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

        self.repository = repository
        self.embedding_client = embedding_client
        self.batch_size = int(batch_size)
        self.index_name = normalized_name
        self.approved_only = bool(approved_only)

    def rebuild(self, *, force: bool = False) -> ItemEmbeddingIndexReport:
        """Incrementally index all eligible rows.

        ``force=True`` re-embeds every eligible item for the active model.
        Otherwise unchanged ``content_hash`` values are skipped.
        """

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
        sources = self.repository.load_indexable_items(
            approved_only=self.approved_only,
            item_ids=item_ids,
        )
        documents: list[ItemEmbeddingDocument] = []
        for source in sources:
            try:
                documents.append(build_item_embedding_document(source.as_mapping()))
            except UnindexableItemDescriptionError:
                continue
        return documents

    def _prune_embeddings(
        self,
        *,
        eligible_item_ids: set[int],
        scope_item_ids: set[int] | None = None,
    ) -> int:
        return self.repository.prune_embeddings(
            embedding_model=self.embedding_client.model,
            eligible_item_ids=eligible_item_ids,
            scope_item_ids=scope_item_ids,
        )

    def _existing_hashes(self, item_ids: list[int] | None = None) -> dict[int, str]:
        return self.repository.existing_hashes(
            embedding_model=self.embedding_client.model,
            item_ids=item_ids,
        )

    def _known_dimension(self) -> int | None:
        return self.repository.known_dimension(embedding_model=self.embedding_client.model)

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
        now = datetime.now(timezone.utc).isoformat()
        self.repository.store_embeddings(
            [
                StoredItemEmbedding(
                    item_id=document.item_id,
                    embedding_model=self.embedding_client.model,
                    embedding_dimension=result.dimension,
                    document_text=document.text,
                    content_hash=document.content_hash,
                    vector=tuple(float(value) for value in vector),
                    updated_at=now,
                )
                for document, vector in zip(documents, result.vectors, strict=True)
            ]
        )

    def _write_state(
        self,
        *,
        dimension: int | None,
        indexed_count: int,
        failed_count: int,
        last_indexed_item_id: int | None,
        last_error: str | None,
    ) -> None:
        self.repository.save_state(
            SemanticIndexState(
                index_name=self.index_name,
                embedding_model=self.embedding_client.model,
                embedding_dimension=dimension,
                indexed_count=indexed_count,
                failed_count=failed_count,
                last_indexed_item_id=last_indexed_item_id,
                last_completed_at=datetime.now(timezone.utc).isoformat(),
                last_error=last_error,
            )
        )

    def _batches(
        self,
        documents: list[ItemEmbeddingDocument],
    ) -> Iterable[list[ItemEmbeddingDocument]]:
        for offset in range(0, len(documents), self.batch_size):
            yield documents[offset : offset + self.batch_size]
