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
    ) -> dict[str, Any]:
        database_record = self._receipt_db.get_receipt_review_record_by_job_id(job_id)
        if database_record and database_record.get("id") is not None:
            try:
                return self._editor.save(
                    int(database_record["id"]),
                    fields=fields,
                    item_corrections=item_corrections,
                    review=review,
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
        approved_path = self._review_service.approved_receipt_path(job_id)
        review_path = self._review_service.review_record_path(job_id)
        approved_path.parent.mkdir(parents=True, exist_ok=True)
        approved_path.write_text(
            json.dumps(approved, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        record = {
            "job_id": job_id,
            "source_receipt": (original_source_path or source_path).name,
            "approved_receipt": approved_path.name,
            "status": approved.get("human_review", {}).get("status"),
            "reviewer": approved.get("human_review", {}).get("reviewer"),
            "notes": approved.get("human_review", {}).get("notes"),
            "reviewed_at": approved.get("human_review", {}).get("reviewed_at"),
            "changed_fields": changed,
            "submitted_fields": fields,
            "submitted_items": item_corrections,
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

        db_import = self._review_service.import_reviewed_receipt(
            job_id,
            approved,
            approved_path,
            original_source_path or source_path,
        )
        record["receipt_db_import"] = db_import
        queue_status = (
            "approved"
            if record.get("status") not in {"rejected", "duplicate_confirmed"}
            else str(record.get("status"))
        )
        self._receipt_db.update_review_status(
            job_id,
            queue_status,
            receipt_db_id=db_import.get("receipt_db_id"),
        )
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
            )
            self._store.add_event(
                job_id,
                {
                    "stage": "human_review",
                    "status": "done",
                    "message": f"Human review saved with {len(changed)} changed field(s).",
                    "details": {
                        "review_status": record.get("status"),
                        "changed_fields": changed,
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
