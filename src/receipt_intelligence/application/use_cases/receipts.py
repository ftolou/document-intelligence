"""Receipt database management use cases."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.application.errors import InvalidRequestError, ResourceNotFoundError
from receipt_intelligence.application.ports.jobs import JobRepository
from receipt_intelligence.application.ports.receipts import (
    ReceiptEditor,
    ReceiptRepository,
    ReviewWorkflow,
)


class ReceiptUseCases:
    def __init__(
        self,
        store: JobRepository,
        receipt_db: ReceiptRepository,
        review_service: ReviewWorkflow,
        editor: ReceiptEditor,
    ) -> None:
        self._store = store
        self._receipt_db = receipt_db
        self._review_service = review_service
        self._editor = editor

    def summary(self) -> dict[str, Any]:
        return self._receipt_db.summary()

    def import_job(self, job_id: str) -> dict[str, Any]:
        if self._store.get(job_id) is None:
            raise ResourceNotFoundError("job not found")

        approved_path = self._review_service.approved_receipt_path(job_id)
        if not approved_path.exists():
            raise InvalidRequestError(
                "receipt must be approved in the Review tab before it can be imported"
            )

        source_path = approved_path
        receipt = self._review_service.read_receipt_json(source_path)
        human_review = (
            receipt.get("human_review") if isinstance(receipt.get("human_review"), dict) else {}
        )
        if str(human_review.get("status") or "") != "approved":
            raise InvalidRequestError(
                "receipt must have human_review.status=approved before import"
            )
        db_import = self._review_service.import_reviewed_receipt(
            job_id,
            receipt,
            source_path,
            source_path,
        )
        self._receipt_db.update_review_status(
            job_id,
            "imported",
            receipt_db_id=db_import.get("receipt_db_id"),
        )
        self._store.update(job_id, receipt_db_import=db_import)
        self._store.add_event(
            job_id,
            {
                "stage": "receipt_db",
                "status": "done",
                "message": (
                    f"Receipt imported into local DB with {db_import['item_count']} item(s)."
                ),
                "details": db_import,
            },
        )
        return {
            "ok": True,
            "job_id": job_id,
            "receipt_db_import": db_import,
            "summary": self._receipt_db.summary(),
        }

    def list_receipts(self, *, limit: int = 200) -> list[dict[str, Any]]:
        return self._receipt_db.list_receipts(limit=limit)

    def get_receipt(self, receipt_id: int) -> dict[str, Any]:
        receipt = self._receipt_db.get_receipt(receipt_id)
        if receipt is None:
            raise ResourceNotFoundError("receipt not found")
        return receipt

    def load_review(self, receipt_id: int) -> dict[str, Any]:
        try:
            return self._editor.load(receipt_id)
        except KeyError as exc:
            raise ResourceNotFoundError("receipt not found") from exc

    def save_review(
        self,
        receipt_id: int,
        *,
        fields: dict[str, Any],
        item_corrections: list[dict[str, Any]],
        review: dict[str, Any],
        identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return self._editor.save(
                receipt_id,
                fields=fields,
                item_corrections=item_corrections,
                review=review,
                identity=identity or {},
            )
        except KeyError as exc:
            raise ResourceNotFoundError("receipt not found") from exc
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

    def delete_receipt(self, receipt_id: str) -> dict[str, Any]:
        try:
            if receipt_id.startswith("job:"):
                result = self._receipt_db.delete_receipt(job_id=receipt_id.split(":", 1)[1])
            else:
                result = self._receipt_db.delete_receipt(receipt_id=int(receipt_id))
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(str(exc)) from exc
        return {"ok": True, "result": result, "summary": self._receipt_db.summary()}

    def delete_all(self, *, confirmation: str, include_review_queue: bool) -> dict[str, Any]:
        if confirmation != "DELETE_ALL_RECEIPTS":
            raise InvalidRequestError("confirmation required: DELETE_ALL_RECEIPTS")
        result = self._receipt_db.delete_all_receipt_data(include_review_queue=include_review_queue)
        return {"ok": True, "deleted": result, "summary": self._receipt_db.summary()}


__all__ = ["ReceiptUseCases"]
