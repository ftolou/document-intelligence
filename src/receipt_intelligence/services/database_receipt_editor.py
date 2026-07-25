"""Database-authoritative editing for already imported receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from receipt_intelligence.services.review_service import ReviewService, apply_human_review
from receipt_intelligence.services.semantic_index_service import (
    ReindexCallback,
    SemanticIndexUpdater,
)
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


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
        self._semantic_index = (
            SemanticIndexUpdater(receipt_db, reindex_callback=reindex_callback)
            if reindex_callback is not None
            else review_service.semantic_index_updater or SemanticIndexUpdater(receipt_db)
        )

    def load(self, receipt_id: int) -> dict[str, Any]:
        receipt = self.receipt_db.get_receipt_edit_document(receipt_id)
        if receipt is None:
            raise KeyError("receipt not found")
        record = self.receipt_db.get_receipt_review_record(receipt_id) or {}
        job_id = str(record.get("job_id") or "").strip()
        image_reference = self.review_service.database_image_reference(record) if job_id else None
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
        database_record = self.receipt_db.get_receipt_review_record(receipt_id) or {}
        job_id = str(
            database_record.get("job_id") or (current.get("_database") or {}).get("job_id") or ""
        ).strip()
        finalization = self.review_service.finalize_human_review(
            job_id,
            updated,
            requested_status=review.get("status"),
        )
        updated = finalization["receipt"]
        database_update = self.receipt_db.update_receipt_from_review(receipt_id, updated)

        previous_status = str(database_update.get("previous_review_status") or "")
        effective_status = str(database_update.get("review_status") or "")
        if effective_status == "approved":
            item_ids = (
                list(database_update.get("all_item_ids") or [])
                if previous_status != "approved"
                else list(database_update.get("semantic_item_ids") or [])
            )
            indexing = self._semantic_index.index_item_ids(item_ids)
        else:
            indexing = {
                "status": "not_required",
                "requested_item_ids": [],
                "message": "Receipt is not approved; semantic embeddings are not created.",
            }

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
                "validation": fresh.get("validation"),
                "review_finalization": {
                    key: value for key, value in finalization.items() if key != "receipt"
                },
                "semantic_index": indexing,
                "artifact_mirror": mirror,
            }
        )
        return payload

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
