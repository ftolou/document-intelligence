"""Backward-compatible facade for the refactored receipt storage layer.

The public ``ReceiptDatabase`` API is intentionally preserved. Internally, Phase
3 separates connection policy, schema migrations, normalization, and focused
repositories so callers can migrate incrementally without a flag-day rewrite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from receipt_intelligence.storage.connection import SQLiteConnectionFactory
from receipt_intelligence.storage.fingerprints import (
    duplicate_score_against_row,
    file_sha256,
    item_overlap,
    receipt_core,
)
from receipt_intelligence.storage.migrations import (
    LATEST_SCHEMA_VERSION,
    MigrationRunner,
)
from receipt_intelligence.storage.models import ReceiptImportResult
from receipt_intelligence.storage.normalization import (
    CATEGORY_ALIASES,
    MERCHANT_ALIASES,
    PARSER_ITEM_TYPES,
    as_float,
    as_str,
    build_item_embedding_text,
    category_from_item,
    extract_item_description,
    first_present,
    normalize_merchant_name,
    normalize_text,
    parser_item_type_from_item,
    tokenize,
    utc_now,
)
from receipt_intelligence.storage.repositories import (
    AnalyticsRepository,
    CatalogRepository,
    ItemRepository,
    ReceiptRepository,
    ReviewRepository,
    SearchRepository,
)
from receipt_intelligence.storage.repositories.base import fts_available

SCHEMA_VERSION = LATEST_SCHEMA_VERSION


class ReceiptDatabase:
    """Compatibility facade over focused SQLite repositories."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.connections = SQLiteConnectionFactory(self.db_path)
        self.migrations = MigrationRunner(self.connections)

        self.items = ItemRepository(self.connections)
        self.receipts = ReceiptRepository(self.connections, self.items)
        self.analytics = AnalyticsRepository(self.connections)
        self.review = ReviewRepository(self.connections)
        self.search = SearchRepository(self.connections)
        self.catalog = CatalogRepository(self.connections)

        self.initialize()

    def connect(self) -> sqlite3.Connection:
        """Return a configured connection for compatibility and diagnostics."""
        return self.connections.connect()

    def initialize(self) -> None:
        self.migrations.migrate()
        self.catalog.seed_product_aliases()

    def receipt_count(self) -> int:
        return self.analytics.receipt_count()

    def item_count(self) -> int:
        return self.analytics.item_count()

    def summary(self) -> dict[str, Any]:
        return self.analytics.summary()

    def import_receipt(
        self,
        *,
        job_id: str,
        receipt: dict[str, Any],
        approved_receipt_path: Path | None = None,
        source_receipt_path: Path | None = None,
        image_path: Path | str | None = None,
    ) -> ReceiptImportResult:
        return self.receipts.import_receipt(
            job_id=job_id,
            receipt=receipt,
            approved_receipt_path=approved_receipt_path,
            source_receipt_path=source_receipt_path,
            image_path=image_path,
        )

    @staticmethod
    def file_sha256(path: Path | str | None) -> str | None:
        return file_sha256(path)

    def _receipt_core(self, receipt: dict[str, Any]) -> dict[str, Any]:
        return receipt_core(receipt)

    @staticmethod
    def _item_overlap(signature_a: str | None, signature_b: str | None) -> float:
        return item_overlap(signature_a, signature_b)

    @staticmethod
    def _duplicate_score_against_row(
        core: dict[str, Any],
        file_hash: str | None,
        row: sqlite3.Row,
    ) -> tuple[float, list[str]]:
        return duplicate_score_against_row(core, file_hash, row)

    def find_duplicate_candidates(
        self,
        *,
        job_id: str,
        receipt: dict[str, Any],
        image_path: Path | str | None = None,
        threshold: float = 70.0,
    ) -> list[dict[str, Any]]:
        return self.review.find_duplicate_candidates(
            job_id=job_id,
            receipt=receipt,
            image_path=image_path,
            threshold=threshold,
        )

    def upsert_review_queue(
        self,
        *,
        job_id: str,
        receipt: dict[str, Any],
        decision: str | None,
        balanced: bool | None,
        difference: float | None,
        issue_count: int | None,
        image_path: Path | str | None,
        final_receipt_path: Path | str | None,
        queue_status: str | None = None,
    ) -> dict[str, Any]:
        return self.review.upsert_review_queue(
            job_id=job_id,
            receipt=receipt,
            decision=decision,
            balanced=balanced,
            difference=difference,
            issue_count=issue_count,
            image_path=image_path,
            final_receipt_path=final_receipt_path,
            queue_status=queue_status,
        )

    def list_review_queue(
        self,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        return self.review.list_review_queue(status=status, limit=limit)

    def update_review_status(
        self,
        job_id: str,
        status: str,
        *,
        receipt_db_id: int | None = None,
    ) -> dict[str, Any]:
        return self.review.update_review_status(
            job_id,
            status,
            receipt_db_id=receipt_db_id,
        )

    def list_receipts(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.receipts.list_receipts(limit=limit)

    def get_receipt(self, receipt_id: int) -> dict[str, Any] | None:
        return self.receipts.get_receipt(receipt_id)

    def get_receipt_edit_document(self, receipt_id: int) -> dict[str, Any] | None:
        return self.receipts.get_receipt_edit_document(receipt_id)

    def update_receipt_from_review(
        self,
        receipt_id: int,
        receipt: dict[str, Any],
    ) -> dict[str, Any]:
        return self.receipts.update_receipt_from_review(receipt_id, receipt)

    def list_receipt_item_ids(self, receipt_id: int) -> list[int]:
        return self.receipts.list_receipt_item_ids(receipt_id)

    def get_receipt_review_record(self, receipt_id: int) -> dict[str, Any] | None:
        return self.receipts.get_receipt_review_record(receipt_id)

    def get_receipt_review_record_by_job_id(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        return self.receipts.get_receipt_review_record_by_job_id(job_id)

    def query_planner_context(self) -> dict[str, Any]:
        return self.analytics.query_planner_context()

    def list_receipts_filtered(
        self,
        *,
        merchant: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 25,
        sort_by: str = "date_desc",
    ) -> list[dict[str, Any]]:
        return self.analytics.list_receipts_filtered(
            merchant=merchant,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            sort_by=sort_by,
        )

    def aggregate_receipts(
        self,
        *,
        merchant: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        aggregation: str = "sum",
        metric: str = "grand_total",
    ) -> dict[str, Any]:
        return self.analytics.aggregate_receipts(
            merchant=merchant,
            date_from=date_from,
            date_to=date_to,
            aggregation=aggregation,
            metric=metric,
        )

    def group_receipts(
        self,
        *,
        group_by: str,
        merchant: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        aggregation: str = "sum",
        metric: str = "grand_total",
        limit: int = 25,
        sort_by: str = "amount_desc",
    ) -> list[dict[str, Any]]:
        return self.analytics.group_receipts(
            group_by=group_by,
            merchant=merchant,
            date_from=date_from,
            date_to=date_to,
            aggregation=aggregation,
            metric=metric,
            limit=limit,
            sort_by=sort_by,
        )

    def delete_receipt(
        self,
        receipt_id: int | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        return self.receipts.delete_receipt(receipt_id=receipt_id, job_id=job_id)

    def delete_all_receipt_data(
        self,
        *,
        include_review_queue: bool = False,
    ) -> dict[str, Any]:
        return self.receipts.delete_all_receipt_data(include_review_queue=include_review_queue)

    def search_items(
        self,
        *,
        semantic_query: str,
        merchant: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        categories: list[str] | None = None,
        limit: int | None = 25,
    ) -> list[dict[str, Any]]:
        return self.search.search_items(
            semantic_query=semantic_query,
            merchant=merchant,
            date_from=date_from,
            date_to=date_to,
            categories=categories,
            limit=limit,
        )

    def expand_query_terms(self, query: str) -> list[str]:
        return self.search.expand_query_terms(query)

    def infer_categories(self, query: str) -> list[str]:
        return self.search.infer_categories(query)

    def _fts_available(self, connection: sqlite3.Connection) -> bool:
        return fts_available(connection)


__all__ = [
    "CATEGORY_ALIASES",
    "MERCHANT_ALIASES",
    "PARSER_ITEM_TYPES",
    "ReceiptDatabase",
    "ReceiptImportResult",
    "SCHEMA_VERSION",
    "as_float",
    "as_str",
    "build_item_embedding_text",
    "category_from_item",
    "extract_item_description",
    "first_present",
    "normalize_merchant_name",
    "normalize_text",
    "parser_item_type_from_item",
    "tokenize",
    "utc_now",
]
