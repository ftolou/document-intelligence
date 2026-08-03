"""Human-review workspace and review-queue use cases."""

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
from receipt_intelligence.receipt_compat import to_review_document


class ReviewUseCases:
    """Coordinate the single review workspace and its canonical database state."""

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

        queue_record = self._receipt_db.get_review_queue_record(job_id)
        receipt = (queue_record or {}).get("receipt")
        source = "review_queue"
        if not isinstance(receipt, dict) or not receipt:
            source_path = self._review_service.preferred_receipt_path(
                job_id,
                stored_approved_path=(database_record or {}).get("approved_receipt_path"),
                stored_source_path=(database_record or {}).get("source_receipt_path"),
            )
            if source_path is None:
                raise ResourceNotFoundError("review draft not found")
            try:
                receipt = self._review_service.read_receipt_json(source_path)
            except Exception as exc:
                raise ResourceNotFoundError(f"could not read receipt JSON: {exc}") from exc
            source = _source_label(source_path)

        job = self._store.get(job_id)
        artifacts = self._review_artifacts(
            job_id,
            job=job,
            database_record=database_record,
        )
        return {
            "job_id": job_id,
            "receipt_db_id": (database_record or {}).get("id"),
            "receipt": to_review_document(receipt),
            "review": self._review_service.load_review_record(job_id, receipt),
            "artifacts": artifacts,
            "receipt_image": artifacts.get("receipt_image"),
            "source": source,
            "editable": True,
            "read_only_reason": None,
            "review_identity": _queue_review_identity(job_id, queue_record),
            "queue_record": _queue_metadata(queue_record),
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

        queue_record = self._receipt_db.get_review_queue_record(job_id)
        legacy_artifact_bootstrap = queue_record is None
        if not legacy_artifact_bootstrap:
            _validate_queue_identity(job_id, queue_record, identity or {})
        receipt = (queue_record or {}).get("receipt")

        stored_source_path = self._review_service.safe_job_artifact_path(
            job_id,
            (database_record or {}).get("source_receipt_path"),
        )
        queued_source_path = self._review_service.safe_job_artifact_path(
            job_id,
            (queue_record or {}).get("final_receipt_path"),
        )
        original_source_path = (
            stored_source_path
            or queued_source_path
            or self._review_service.final_receipt_path(job_id)
        )
        if not isinstance(receipt, dict) or not receipt:
            source_path = self._review_service.preferred_receipt_path(
                job_id,
                stored_approved_path=(database_record or {}).get("approved_receipt_path"),
                stored_source_path=(database_record or {}).get("source_receipt_path"),
            )
            if source_path is None:
                raise ResourceNotFoundError("review draft not found")
            receipt = self._review_service.read_receipt_json(source_path)
            original_source_path = original_source_path or source_path

        if legacy_artifact_bootstrap:
            bootstrap_path = original_source_path or self._review_service.approved_receipt_path(
                job_id
            )
            self._review_service.sync_review_queue(
                job_id,
                receipt,
                receipt_path=bootstrap_path,
                queue_status="needs_review",
            )

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
            "source_receipt": original_source_path.name if original_source_path else None,
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
                original_source_path or approved_path,
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
        revision = self._receipt_db.save_review_revision(
            job_id=job_id,
            receipt=approved,
            requested_status=finalization.get("requested_status"),
            effective_status=queue_status,
            reviewer=human_review.get("reviewer"),
            notes=human_review.get("notes"),
            changed_fields=changed,
            receipt_db_id=db_import.get("receipt_db_id"),
            expected_revision=(
                0
                if legacy_artifact_bootstrap
                else int((identity or {}).get("review_revision") or 0)
            ),
        )
        review_queue.update(revision)
        record["review_queue"] = review_queue
        record["review_revision"] = revision.get("revision")
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
                        f"Human review revision {revision.get('revision')} saved with "
                        f"{len(changed)} changed field(s); effective status is "
                        f"{record.get('status')}."
                    ),
                    "details": {
                        "requested_status": record.get("requested_status"),
                        "review_status": record.get("status"),
                        "review_revision": revision.get("revision"),
                        "approval_blocked": record.get("approval_blocked"),
                        "import_decision": (record.get("validation") or {}).get("import_decision"),
                        "changed_fields": changed,
                        "semantic_index_status": semantic_index.get("status"),
                    },
                },
            )
        response_source = "review_queue"
        response_identity: dict[str, Any] = {
            "source": "review_queue",
            "job_id": job_id,
            "review_revision": revision.get("revision"),
        }
        response_receipt = to_review_document(approved)
        imported_receipt_id = db_import.get("receipt_db_id")
        if imported_receipt_id is not None:
            try:
                loaded = self._editor.load(int(imported_receipt_id))
                response_source = str(loaded.get("source") or "database")
                response_identity = loaded.get("review_identity") or response_identity
                response_receipt = loaded.get("receipt") or response_receipt
                artifacts.update(loaded.get("artifacts") or {})
            except KeyError:
                pass

        return {
            "ok": True,
            "job_id": job_id,
            "receipt_db_id": imported_receipt_id,
            "review": record,
            "receipt": response_receipt,
            "artifacts": artifacts,
            "receipt_db_import": db_import,
            "validation": response_receipt.get("validation"),
            "review_finalization": {
                key: value for key, value in finalization.items() if key != "receipt"
            },
            "semantic_index": semantic_index,
            "review_queue": review_queue,
            "review_identity": response_identity,
            "source": response_source,
            "editable": True,
        }

    def list_queue(self, *, status: str = "all", limit: int = 200) -> list[dict[str, Any]]:
        return self._receipt_db.list_review_queue(status=status, limit=limit)

    def queue_summary(self) -> dict[str, Any]:
        return self._receipt_db.review_queue_summary()

    def update_queue_status(self, job_id: str, status: str) -> dict[str, Any]:
        status_value = str(status or "").strip()
        allowed = {"duplicate_confirmed", "dismissed_duplicate"}
        if status_value not in allowed:
            raise InvalidRequestError(
                "Approve, reject, and draft decisions must be saved through the Review workspace."
            )
        try:
            result = self._receipt_db.update_review_status(job_id, status_value)
        except (KeyError, ValueError) as exc:
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


def _queue_review_identity(
    job_id: str,
    queue_record: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "source": "review_queue",
        "job_id": str(job_id),
        "review_revision": int((queue_record or {}).get("review_revision") or 0),
    }


def _validate_queue_identity(
    job_id: str,
    queue_record: dict[str, Any] | None,
    identity: dict[str, Any],
) -> None:
    if queue_record is None:
        raise ResourceNotFoundError("review queue record not found")
    expected = _queue_review_identity(job_id, queue_record)
    if str(identity.get("source") or "") != "review_queue":
        raise InvalidRequestError("review identity is required; reload the receipt before saving")
    if str(identity.get("job_id") or "") != expected["job_id"]:
        raise InvalidRequestError("review job identity does not match; reload the receipt")
    try:
        submitted_revision = int(identity.get("review_revision"))
    except (TypeError, ValueError):
        submitted_revision = -1
    if submitted_revision != expected["review_revision"]:
        raise InvalidRequestError("review state is stale; reload the receipt before saving")


def _queue_metadata(queue_record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not queue_record:
        return None
    keys = (
        "job_id",
        "queue_status",
        "decision",
        "balanced",
        "difference",
        "issue_count",
        "duplicate_status",
        "duplicate_score",
        "duplicate_candidates",
        "reason_codes",
        "review_revision",
        "reviewer",
        "review_notes",
        "reviewed_at",
        "updated_at",
    )
    return {key: queue_record.get(key) for key in keys}


def _source_label(path: Path) -> str:
    return "approved_receipt" if path.name == "approved_receipt.json" else "final_receipt"


__all__ = ["ReviewUseCases"]
