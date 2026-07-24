"""Application use cases exposed to Flask blueprints."""

from __future__ import annotations

from dataclasses import dataclass

from flask import Flask, current_app

import receipt_intelligence.settings as settings
from receipt_intelligence.application.ports import OcrEngine
from receipt_intelligence.application.use_cases.jobs import JobUseCases
from receipt_intelligence.application.use_cases.query import AskReceipts, ReceiptQueryExecutor
from receipt_intelligence.application.use_cases.receipts import ReceiptUseCases
from receipt_intelligence.application.use_cases.reviews import ReviewUseCases
from receipt_intelligence.application.use_cases.runtime import RuntimeUseCases
from receipt_intelligence.composition import build_ocr_engine
from receipt_intelligence.observability.query import QueryTelemetrySink
from receipt_intelligence.runtime.paths import RuntimePaths
from receipt_intelligence.services.database_receipt_editor import DatabaseReceiptEditor
from receipt_intelligence.services.job_processing import JobProcessingService
from receipt_intelligence.services.review_service import ReviewService, apply_human_review
from receipt_intelligence.services.runtime_information import RuntimeInformationService
from receipt_intelligence.storage.job_store import JobStore
from receipt_intelligence.storage.receipt_db import ReceiptDatabase

_EXTENSION_KEY = "receipt_intelligence.application"


@dataclass(frozen=True, slots=True)
class AppServices:
    jobs: JobUseCases
    receipts: ReceiptUseCases
    reviews: ReviewUseCases
    ask_receipts: AskReceipts
    runtime: RuntimeUseCases


def init_app_services(
    app: Flask,
    *,
    job_store: JobStore | None = None,
    receipt_db: ReceiptDatabase | None = None,
    query_telemetry: QueryTelemetrySink | None = None,
    receipt_query_service: ReceiptQueryExecutor | None = None,
    runtime_paths: RuntimePaths | None = None,
    ocr_engine: OcrEngine | None = None,
) -> AppServices:
    resolved_paths = runtime_paths or settings.RUNTIME_PATHS
    telemetry_path = (
        resolved_paths.logs_dir / "query_events.jsonl"
        if runtime_paths is not None
        else settings.QUERY_TELEMETRY_PATH
    )
    resolved_store = job_store or JobStore(settings.RESULTS_DIR)
    resolved_database = receipt_db or ReceiptDatabase(settings.RECEIPT_DB_PATH)
    resolved_telemetry = query_telemetry or QueryTelemetrySink.from_path(
        telemetry_path,
        enabled=settings.QUERY_TELEMETRY_ENABLED,
    )
    if receipt_query_service is None:
        from receipt_intelligence.rag_sql.application import (
            build_receipt_query_service_from_settings,
        )

        resolved_query_service = build_receipt_query_service_from_settings(
            telemetry_sink=resolved_telemetry,
        )
    else:
        resolved_query_service = receipt_query_service
    resolved_review_service = ReviewService(resolved_store, resolved_database)
    resolved_editor = DatabaseReceiptEditor(resolved_database, resolved_review_service)
    processor = JobProcessingService(
        resolved_store,
        resolved_database,
        ocr_engine=ocr_engine or build_ocr_engine(),
    )
    services = AppServices(
        jobs=JobUseCases(resolved_store, processor),
        receipts=ReceiptUseCases(
            resolved_store,
            resolved_database,
            resolved_review_service,
            resolved_editor,
        ),
        reviews=ReviewUseCases(
            resolved_store,
            resolved_database,
            resolved_review_service,
            resolved_editor,
            apply_human_review,
        ),
        ask_receipts=AskReceipts(resolved_query_service),
        runtime=RuntimeUseCases(
            RuntimeInformationService(resolved_database, resolved_paths)
        ),
    )
    app.extensions[_EXTENSION_KEY] = services
    return services


def get_app_services() -> AppServices:
    services = current_app.extensions.get(_EXTENSION_KEY)
    if not isinstance(services, AppServices):
        raise RuntimeError("Receipt Intelligence application use cases are not initialized.")
    return services


__all__ = ["AppServices", "get_app_services", "init_app_services"]
