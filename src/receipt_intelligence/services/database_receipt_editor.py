"""Database-authoritative editing for already imported receipts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import receipt_intelligence.settings as settings
from receipt_intelligence.adapters.storage.sqlite.semantic_index import (
    SQLiteSemanticIndexRepository,
)
from receipt_intelligence.rag.embedding_client import OllamaEmbeddingClient
from receipt_intelligence.rag.item_indexer import ItemEmbeddingIndexer
from receipt_intelligence.services.review_service import ReviewService, apply_human_review
from receipt_intelligence.storage.receipt_db import ReceiptDatabase

ReindexCallback = Callable[[list[int]], dict[str, Any]]


class DatabaseReceiptEditor:
    """Edit persistent receipts by ``receipt_id`` without requiring job artifacts.

    SQLite is authoritative. Existing approved JSON is updated only as a best-effort
    mirror after the database transaction has committed.
    """

    def __init__(
        self,
        receipt_db: ReceiptDatabase,
        review_service: ReviewService,
        *,
        reindex_callback: ReindexCallback | None = None,
    ) -> None:
        self.receipt_db = receipt_db
        self.review_service = review_service
        self._reindex_callback = reindex_callback

    def load(self, receipt_id: int) -> dict[str, Any]:
        receipt = self.receipt_db.get_receipt_edit_document(receipt_id)
        if receipt is None:
            raise KeyError("receipt not found")
        record = self.receipt_db.get_receipt_review_record(receipt_id) or {}
        job_id = str(record.get("job_id") or "").strip()
        image_reference = (
            self.review_service.database_image_reference(record) if job_id else None
        )
        artifacts = {"receipt_image": image_reference} if image_reference else {}
        review = (
            receipt.get("human_review") if isinstance(receipt.get("human_review"), dict) else None
        )
        return {
            "receipt_id": int(receipt_id),
            "receipt_db_id": int(receipt_id),
            "job_id": job_id or None,
            "receipt": receipt,
            "review": review,
            "artifacts": artifacts,
            "receipt_image": image_reference,
            "source": "database",
            "editable": True,
            "read_only_reason": None,
        }

    def save(
        self,
        receipt_id: int,
        *,
        fields: dict[str, Any],
        item_corrections: list[dict[str, Any]],
        review: dict[str, Any],
    ) -> dict[str, Any]:
        current = self.receipt_db.get_receipt_edit_document(receipt_id)
        if current is None:
            raise KeyError("receipt not found")

        updated, changed_fields = apply_human_review(
            current,
            fields,
            item_corrections,
            review,
        )
        database_update = self.receipt_db.update_receipt_from_review(receipt_id, updated)
        semantic_item_ids = list(database_update.get("semantic_item_ids") or [])
        indexing = self._reindex(semantic_item_ids)

        fresh = self.receipt_db.get_receipt_edit_document(receipt_id)
        if fresh is None:
            raise RuntimeError("receipt disappeared after database update")
        mirror = self._write_approved_json_mirror(receipt_id, fresh)
        payload = self.load(receipt_id)
        payload.update(
            {
                "ok": True,
                "receipt": fresh,
                "review": {
                    **(
                        fresh.get("human_review")
                        if isinstance(fresh.get("human_review"), dict)
                        else {}
                    ),
                    "changed_fields": changed_fields,
                },
                "database_update": database_update,
                "receipt_db_import": {
                    "receipt_db_id": int(receipt_id),
                    "job_id": database_update.get("job_id"),
                    "item_count": database_update.get("item_count", 0),
                    "updated_at": database_update.get("updated_at"),
                },
                "semantic_index": indexing,
                "artifact_mirror": mirror,
            }
        )
        return payload

    def _reindex(self, item_ids: list[int]) -> dict[str, Any]:
        if not item_ids:
            return {
                "status": "not_required",
                "requested_item_ids": [],
                "message": "No embedded semantic fields changed.",
            }
        if self._reindex_callback is not None:
            try:
                return self._reindex_callback(item_ids)
            except Exception as exc:
                return {
                    "status": "failed",
                    "requested_item_ids": item_ids,
                    "error": f"{type(exc).__name__}: {exc}",
                    "message": (
                        "Database changes were saved. Semantic reindexing can be retried later."
                    ),
                }
        if not settings.RAG_EMBEDDING_ENABLED:
            return {
                "status": "pending",
                "requested_item_ids": item_ids,
                "message": "Semantic embeddings are disabled; stale vectors were removed.",
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
                ).index_item_ids(item_ids)
            status = "current" if report.failed == 0 else "failed"
            return {
                "status": status,
                "requested_item_ids": item_ids,
                "report": report.model_dump(mode="json"),
                "message": (
                    "Semantic index updated."
                    if status == "current"
                    else "Database changes were saved, but one or more embeddings failed."
                ),
            }
        except Exception as exc:
            return {
                "status": "failed",
                "requested_item_ids": item_ids,
                "error": f"{type(exc).__name__}: {exc}",
                "message": (
                    "Database changes were saved. Semantic reindexing can be retried later."
                ),
            }

    def _write_approved_json_mirror(
        self,
        receipt_id: int,
        receipt: dict[str, Any],
    ) -> dict[str, Any]:
        record = self.receipt_db.get_receipt_review_record(receipt_id) or {}
        job_id = str(record.get("job_id") or "").strip()
        if not job_id:
            return {"status": "not_available"}
        approved_path = self.review_service.safe_job_artifact_path(
            job_id,
            record.get("approved_receipt_path"),
        )
        if approved_path is None:
            approved_path = self.review_service.safe_job_artifact_path(
                job_id,
                self.review_service.approved_receipt_path(job_id),
            )
        if approved_path is None:
            return {"status": "not_available"}

        try:
            mirror = json.loads(json.dumps(receipt, ensure_ascii=False, default=str))
            mirror.pop("_database", None)
            items = mirror.get("items") if isinstance(mirror.get("items"), list) else []
            for item in items:
                if isinstance(item, dict):
                    item.pop("_db_item_id", None)
            Path(approved_path).write_text(
                json.dumps(mirror, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            return {"status": "updated"}
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }


__all__ = ["DatabaseReceiptEditor"]
