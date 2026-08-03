"""Database-authoritative editing for already imported receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from receipt_intelligence.receipt_compat import to_review_document
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
            "receipt": to_review_document(receipt),
            "review": review,
            "artifacts": artifacts,
            "receipt_image": image_reference,
            "source": "database",
            "editable": True,
            "read_only_reason": None,
            "review_identity": _review_identity(receipt_id, receipt, record),
        }

    def save(
        self,
        receipt_id: int,
        *,
        fields: dict[str, Any],
        item_corrections: list[dict[str, Any]],
        review: dict[str, Any],
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        current = self.receipt_db.get_receipt_edit_document(receipt_id)
        if current is None:
            raise KeyError("receipt not found")

        database_record = self.receipt_db.get_receipt_review_record(receipt_id) or {}
        expected_identity = _validate_review_identity(
            receipt_id,
            current,
            database_record,
            item_corrections,
            identity,
        )

        updated, changed_fields = apply_human_review(
            current,
            fields,
            item_corrections,
            review,
        )
        job_id = str(
            database_record.get("job_id") or (current.get("_database") or {}).get("job_id") or ""
        ).strip()
        finalization = self.review_service.finalize_human_review(
            job_id,
            updated,
            requested_status=review.get("status"),
        )
        updated = finalization["receipt"]
        database_update = self.receipt_db.update_receipt_from_review(
            receipt_id,
            updated,
            expected_job_id=expected_identity["job_id"],
            expected_updated_at=expected_identity["updated_at"],
        )

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
        approved_path = self.review_service.approved_receipt_path(job_id)
        review_queue = self.review_service.sync_review_queue(
            job_id,
            fresh,
            receipt_path=approved_path,
            queue_status=effective_status or "needs_review",
            receipt_db_id=int(receipt_id),
        )
        human_review = (
            fresh.get("human_review") if isinstance(fresh.get("human_review"), dict) else {}
        )
        revision = self.receipt_db.save_review_revision(
            job_id=job_id,
            receipt=fresh,
            requested_status=finalization.get("requested_status"),
            effective_status=effective_status or "needs_review",
            reviewer=human_review.get("reviewer"),
            notes=human_review.get("notes"),
            changed_fields=changed_fields,
            receipt_db_id=int(receipt_id),
        )
        review_queue.update(revision)
        payload = self.load(receipt_id)
        payload.update(
            {
                "ok": True,
                "receipt": to_review_document(fresh),
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
                "review_queue": review_queue,
                "review_revision": revision,
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


def _review_identity(
    receipt_id: int,
    receipt: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    database = receipt.get("_database") if isinstance(receipt.get("_database"), dict) else {}
    items = receipt.get("items") if isinstance(receipt.get("items"), list) else []
    item_ids = [
        int(item["_db_item_id"])
        for item in items
        if isinstance(item, dict) and item.get("_db_item_id") not in (None, "")
    ]
    return {
        "source": "database",
        "receipt_id": int(receipt_id),
        "job_id": str(record.get("job_id") or database.get("job_id") or ""),
        "updated_at": str(record.get("updated_at") or database.get("updated_at") or ""),
        "item_ids": item_ids,
    }


def _validate_review_identity(
    receipt_id: int,
    current: dict[str, Any],
    record: dict[str, Any],
    item_corrections: list[dict[str, Any]],
    identity: dict[str, Any],
) -> dict[str, Any]:
    expected = _review_identity(receipt_id, current, record)
    if not isinstance(identity, dict) or not identity:
        raise ValueError("review identity is required; reload the receipt before saving")
    if str(identity.get("source") or "") != "database":
        raise ValueError("review identity source does not match the database receipt")
    try:
        submitted_receipt_id = int(identity.get("receipt_id"))
    except (TypeError, ValueError):
        submitted_receipt_id = -1
    if submitted_receipt_id != expected["receipt_id"]:
        raise ValueError("review receipt identity does not match; reload the receipt")
    if str(identity.get("job_id") or "") != expected["job_id"]:
        raise ValueError("review job identity does not match; reload the receipt")
    if str(identity.get("updated_at") or "") != expected["updated_at"]:
        raise ValueError("review state is stale; reload the receipt before saving")

    try:
        submitted_item_ids = [int(value) for value in identity.get("item_ids") or []]
    except (TypeError, ValueError):
        submitted_item_ids = []
    if submitted_item_ids != expected["item_ids"]:
        raise ValueError("review item identities do not match; reload the receipt")

    if len(item_corrections) != len(expected["item_ids"]):
        raise ValueError("review must submit every existing database item row")
    seen_indexes: set[int] = set()
    seen_item_ids: set[int] = set()
    for correction in item_corrections:
        if not isinstance(correction, dict):
            raise ValueError("each item correction must be an object")
        try:
            index = int(correction.get("index"))
            item_id = int(correction.get("item_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("each database item correction requires index and item_id") from exc
        if index < 0 or index >= len(expected["item_ids"]):
            raise ValueError("review item index is invalid; reload the receipt")
        if index in seen_indexes or item_id in seen_item_ids:
            raise ValueError("review contains duplicate item identities")
        if expected["item_ids"][index] != item_id:
            raise ValueError("review item identity does not match its receipt row")
        seen_indexes.add(index)
        seen_item_ids.add(item_id)
    return expected
