"""Ports required by receipt and human-review use cases."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from receipt_intelligence.application.ports.jobs import JobRepository

ReviewApplier = Callable[
    [dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]],
    tuple[dict[str, Any], list[str]],
]


class ReceiptRepository(Protocol):
    def summary(self) -> dict[str, Any]: ...

    def update_review_status(
        self,
        job_id: str,
        status: str,
        *,
        receipt_db_id: int | None = None,
    ) -> dict[str, Any]: ...

    def list_receipts(self, limit: int = 200) -> list[dict[str, Any]]: ...

    def get_receipt(self, receipt_id: int) -> dict[str, Any] | None: ...

    def delete_receipt(
        self,
        *,
        receipt_id: int | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]: ...

    def delete_all_receipt_data(self, *, include_review_queue: bool) -> dict[str, Any]: ...

    def get_receipt_review_record_by_job_id(self, job_id: str) -> dict[str, Any] | None: ...

    def list_review_queue(
        self,
        *,
        status: str = "all",
        limit: int = 200,
    ) -> list[dict[str, Any]]: ...


class ReviewWorkflow(Protocol):
    def approved_receipt_path(self, job_id: str) -> Path: ...

    def final_receipt_path(self, job_id: str) -> Path | None: ...

    def review_record_path(self, job_id: str) -> Path: ...

    def safe_job_artifact_path(
        self,
        job_id: str,
        value: str | Path | None,
    ) -> Path | None: ...

    def preferred_receipt_path(
        self,
        job_id: str,
        *,
        stored_approved_path: str | Path | None = None,
        stored_source_path: str | Path | None = None,
    ) -> Path | None: ...

    def read_receipt_json(self, path: Path) -> dict[str, Any]: ...

    def load_review_record(
        self,
        job_id: str,
        receipt: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...

    def database_image_reference(self, record: dict[str, Any]) -> dict[str, str] | None: ...

    def review_image_reference(self, job_id: str) -> dict[str, str] | None: ...

    def import_reviewed_receipt(
        self,
        job_id: str,
        receipt: dict[str, Any],
        approved_path: Path,
        source_path: Path | None = None,
    ) -> dict[str, Any]: ...

    def finalize_human_review(
        self,
        job_id: str,
        receipt: dict[str, Any],
        *,
        requested_status: str | None,
    ) -> dict[str, Any]: ...

    def sync_review_queue(
        self,
        job_id: str,
        receipt: dict[str, Any],
        *,
        receipt_path: Path,
        queue_status: str,
        receipt_db_id: int | None = None,
    ) -> dict[str, Any]: ...

    def index_receipt_items(self, receipt_id: int) -> dict[str, Any]: ...

    def index_item_ids(self, item_ids: list[int]) -> dict[str, Any]: ...


class ReceiptEditor(Protocol):
    def load(self, receipt_id: int) -> dict[str, Any]: ...

    def save(
        self,
        receipt_id: int,
        *,
        fields: dict[str, Any],
        item_corrections: list[dict[str, Any]],
        review: dict[str, Any],
    ) -> dict[str, Any]: ...


__all__ = [
    "JobRepository",
    "ReceiptEditor",
    "ReceiptRepository",
    "ReviewApplier",
    "ReviewWorkflow",
]
