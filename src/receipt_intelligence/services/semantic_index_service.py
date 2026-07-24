"""Best-effort semantic-index updates after durable receipt changes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import receipt_intelligence.settings as settings
from receipt_intelligence.adapters.storage.sqlite.semantic_index import (
    SQLiteSemanticIndexRepository,
)
from receipt_intelligence.rag.embedding_client import OllamaEmbeddingClient
from receipt_intelligence.rag.item_indexer import ItemEmbeddingIndexer
from receipt_intelligence.storage.receipt_db import ReceiptDatabase

ReindexCallback = Callable[[list[int]], dict[str, Any]]


class SemanticIndexUpdater:
    """Index selected SQL item rows without coupling approval to provider uptime.

    Receipt/database changes are committed before this service is called. Every
    provider or storage failure is converted into a result object so a failed
    embedding request never rolls back an approved human review.
    """

    def __init__(
        self,
        receipt_db: ReceiptDatabase,
        *,
        reindex_callback: ReindexCallback | None = None,
    ) -> None:
        self.receipt_db = receipt_db
        self._reindex_callback = reindex_callback

    def index_item_ids(self, item_ids: Iterable[int]) -> dict[str, Any]:
        selected = sorted({int(item_id) for item_id in item_ids if int(item_id) > 0})
        if not selected:
            return {
                "status": "not_required",
                "requested_item_ids": [],
                "message": "No eligible receipt items require semantic indexing.",
            }

        if self._reindex_callback is not None:
            try:
                result = self._reindex_callback(selected)
                payload = dict(result) if isinstance(result, dict) else {"status": "current"}
                payload.setdefault("status", "current")
                payload.setdefault("requested_item_ids", selected)
                return payload
            except Exception as exc:
                return self._failure(selected, exc)

        if not settings.RAG_EMBEDDING_ENABLED:
            return {
                "status": "pending",
                "requested_item_ids": selected,
                "message": "Semantic embeddings are disabled; indexing can be retried later.",
            }

        try:
            with OllamaEmbeddingClient(
                base_url=settings.OLLAMA_URL,
                model=settings.RAG_EMBEDDING_MODEL,
                timeout_seconds=settings.RAG_EMBEDDING_TIMEOUT_SECONDS,
                keep_alive=settings.RAG_EMBEDDING_KEEP_ALIVE,
            ) as embedding_client:
                report = ItemEmbeddingIndexer(
                    repository=SQLiteSemanticIndexRepository(self.receipt_db.db_path),
                    embedding_client=embedding_client,
                    batch_size=settings.RAG_EMBEDDING_BATCH_SIZE,
                ).index_item_ids(selected)
            status = "current" if report.failed == 0 else "failed"
            return {
                "status": status,
                "requested_item_ids": selected,
                "report": report.model_dump(mode="json"),
                "message": (
                    "Semantic index updated."
                    if status == "current"
                    else "Receipt changes were saved, but one or more embeddings failed."
                ),
            }
        except Exception as exc:
            return self._failure(selected, exc)

    @staticmethod
    def _failure(item_ids: list[int], exc: Exception) -> dict[str, Any]:
        return {
            "status": "failed",
            "requested_item_ids": item_ids,
            "error": f"{type(exc).__name__}: {exc}",
            "message": "Receipt changes were saved. Semantic indexing can be retried later.",
        }


__all__ = ["ReindexCallback", "SemanticIndexUpdater"]
