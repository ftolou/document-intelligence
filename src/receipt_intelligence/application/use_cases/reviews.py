"""Human-review and review-queue use cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from receipt_intelligence.application.errors import InvalidRequestError, ResourceNotFoundError
from receipt_intelligence.application.ports.jobs import JobRepository
from receipt_intelligence.application.ports.receipts import (
    ReceiptEditor,
    ReceiptRepository,
    ReviewApplier,
    ReviewWorkflow,
)
from receipt_intelligence.application.resources import artifact_reference


class ReviewUseCases:
    def __init__(
        self,
        store: JobRepository,
        receipt_db: ReceiptRepository,
        review_service: ReviewWorkflow,
        editor: ReceiptEditor,
        review_applier: ReviewApplier,
    ) -> None:
        self._store = store
        self._receipt_db = receipt_db
        self._review_service = review_service
        self._editor = editor
        self._review_applier = review_applier

    def get_review(self, job_id: str) -> dict[str, Any]:
        database_record = self._receipt_db.get_receipt_review_record_by_job_id(job_id)
        if database_record and database_record.get("id") is not None:
            try:
                return self._editor.load(int(database_record["id"]))
            except KeyError as exc:
                raise ResourceNotFoundError("receipt not found") from exc

        source_path = self._review_service.preferred_receipt_path(
            job_id,
            stored_approved_path=(database_record or {}).get("approved_receipt_path"),
            stored_source_path=(database_record or {}).get("source_receipt_path"),
        )
        if source_path is None:
            raise ResourceNotFoundError("approved/final receipt artifact not found")

        try:
            receipt = self._review_service.read_receipt_json(source_path)
        except Exception as exc:
            raise ResourceNotFoundError(f"could not read receipt JSON: {exc}") from exc

        job = self._store.get(job_id)
        artifacts = self._review_artifacts(
            job_id,
            job=job,
            database_record=database_record,
        )
        return {
            "job_id": job_id,
            "receipt_db_id": (database_record or {}).get("id"),
            "receipt": receipt,
            "review": self._review_service.load_review_record(job_id, receipt),
            "artifacts": artifacts,
            "receipt_image": artifacts.get("receipt_image"),
            "source": _source_label(source_path),
            "editable": True,
            "read_only_reason": None,
        }

    def save_review(
        self,
        job_id: str,
        *,
        fields: dict[str, Any],
        item_corrections: list[dict[str, Any]],
        review: dict[str, Any],
        identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        database_record = self._receipt_db.get_receipt_review_record_by_job_id(job_id)
        if database_record and database_record.get("id") is not None:
            try:
                return self._editor.save(
                    int(database_record["id"]),
                    fields=fields,
                    item_corrections=item_corrections,
                    review=review,
                    identity=identity or {},
                )
            except KeyError as exc:
                raise ResourceNotFoundError("receipt not found") from exc
            except ValueError as exc:
                raise InvalidRequestError(str(exc)) from exc

        stored_source_path = self._review_service.safe_job_artifact_path(
            job_id,
            (database_record or {}).get("source_receipt_path"),
        )
        original_source_path = stored_source_path or self._review_service.final_receipt_path(job_id)
        source_path = self._review_service.preferred_receipt_path(
            job_id,
            stored_approved_path=(database_record or {}).get("approved_receipt_path"),
            stored_source_path=(database_record or {}).get("source_receipt_path"),
        )
        if source_path is None:
            raise ResourceNotFoundError("approved/final receipt artifact not found")

        receipt = self._review_service.read_receipt_json(source_path)
        approved, changed = self._review_applier(
            receipt,
            fields,
            item_corrections,
            review,
        )
        try:
            finalization = self._review_service.finalize_human_review(
                job_id,
                approved,
                requested_status=review.get("status"),
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        approved = finalization["receipt"]
        approved_path = self._review_service.approved_receipt_path(job_id)
        review_path = self._review_service.review_record_path(job_id)
        approved_path.parent.mkdir(parents=True, exist_ok=True)
        approved_path.write_text(
            json.dumps(approved, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        human_review = (
            approved.get("human_review") if isinstance(approved.get("human_review"), dict) else {}
        )
        record = {
            "job_id": job_id,
            "source_receipt": (original_source_path or source_path).name,
            "approved_receipt": approved_path.name,
            "requested_status": finalization.get("requested_status"),
            "status": human_review.get("status"),
            "reviewer": human_review.get("reviewer"),
            "notes": human_review.get("notes"),
            "reviewed_at": human_review.get("reviewed_at"),
            "changed_fields": changed,
            "submitted_fields": fields,
            "submitted_items": item_corrections,
            "validation": finalization.get("validation"),
            "approval_blocked": finalization.get("approval_blocked", False),
            "validation_override": finalization.get("validation_override", False),
        }
        review_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        job = self._store.get(job_id)
        if job is not None:
            self._store.register_artifact(
                job_id, "approved_receipt", approved_path, category="review"
            )
            self._store.register_artifact(job_id, "human_review", review_path, category="review")

        if finalization.get("import_allowed"):
            db_import = self._review_service.import_reviewed_receipt(
                job_id,
                approved,
                approved_path,
                original_source_path or source_path,
            )
            semantic_index = self._review_service.index_receipt_items(
                int(db_import["receipt_db_id"])
            )
        else:
            db_import = {
                "status": "not_imported",
                "receipt_db_id": None,
                "job_id": job_id,
                "item_count": 0,
                "reason": (
                    "approval_blocked"
                    if finalization.get("approval_blocked")
                    else f"review_status_{finalization.get('effective_status')}"
                ),
            }
            semantic_index = {
                "status": "not_required",
                "requested_item_ids": [],
                "message": "Receipt is not approved for import; embeddings were not created.",
            }
        record["receipt_db_import"] = db_import
        record["semantic_index"] = semantic_index
        queue_status = str(finalization.get("queue_status") or "needs_review")
        review_queue = self._review_service.sync_review_queue(
            job_id,
            approved,
            receipt_path=approved_path,
            queue_status=queue_status,
            receipt_db_id=db_import.get("receipt_db_id"),
        )
        record["review_queue"] = review_queue
        review_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if job is not None:
            self._store.register_artifact(job_id, "human_review", review_path, category="review")

        artifacts = self._review_artifacts(
            job_id,
            job=job,
            database_record=self._receipt_db.get_receipt_review_record_by_job_id(job_id),
        )
        artifacts["approved_receipt"] = artifact_reference(job_id, approved_path)
        artifacts["human_review"] = artifact_reference(job_id, review_path)
        if job is not None:
            result = job.get("result") if isinstance(job.get("result"), dict) else {}
            result["artifacts"] = artifacts
            self._store.update(
                job_id,
                result=result,
                review=record,
                receipt_db_import=db_import,
                semantic_index=semantic_index,
                review_queue_status=queue_status,
            )
            self._store.add_event(
                job_id,
                {
                    "stage": "human_review",
                    "status": "done",
                    "message": (
                        f"Human review saved with {len(changed)} changed field(s); "
                        f"effective status is {record.get('status')}."
                    ),
                    "details": {
                        "requested_status": record.get("requested_status"),
                        "review_status": record.get("status"),
                        "approval_blocked": record.get("approval_blocked"),
                        "import_decision": (record.get("validation") or {}).get("import_decision"),
                        "changed_fields": changed,
                        "semantic_index_status": semantic_index.get("status"),
                    },
                },
            )
        return {
            "ok": True,
            "job_id": job_id,
            "receipt_db_id": db_import.get("receipt_db_id"),
            "review": record,
            "receipt": approved,
            "artifacts": artifacts,
            "receipt_db_import": db_import,
            "validation": approved.get("validation"),
            "review_finalization": {
                key: value for key, value in finalization.items() if key != "receipt"
            },
            "semantic_index": semantic_index,
            "review_queue": review_queue,
            "source": "approved_receipt",
            "editable": True,
        }

    def list_queue(self, *, status: str = "all", limit: int = 200) -> list[dict[str, Any]]:
        return self._receipt_db.list_review_queue(status=status, limit=limit)

    def update_queue_status(self, job_id: str, status: str) -> dict[str, Any]:
        status_value = str(status or "").strip()
        if not status_value:
            raise InvalidRequestError("missing status")
        try:
            result = self._receipt_db.update_review_status(job_id, status_value)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        self._store.update(job_id, review_queue_status=status_value)
        self._store.add_event(
            job_id,
            {
                "stage": "review_queue",
                "status": "done",
                "message": f"Review queue status changed to {status_value}.",
                "details": result,
            },
        )
        return {"ok": True, "result": result}

    def _review_artifacts(
        self,
        job_id: str,
        *,
        job: dict[str, Any] | None,
        database_record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result = (
            job.get("result")
            if isinstance(job, dict) and isinstance(job.get("result"), dict)
            else {}
        )
        existing = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
        artifacts = dict(existing)
        image_reference = None
        if database_record:
            image_reference = self._review_service.database_image_reference(database_record)
        if not image_reference:
            image_reference = self._review_service.review_image_reference(job_id)
        if image_reference:
            artifacts["receipt_image"] = image_reference
        return artifacts


def _source_label(path: Path) -> str:
    return "approved_receipt" if path.name == "approved_receipt.json" else "final_receipt"


__all__ = ["ReviewUseCases"]
