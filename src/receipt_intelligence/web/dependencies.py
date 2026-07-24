"""Application use cases exposed to Flask blueprints."""

from __future__ import annotations

import atexit
from dataclasses import dataclass

from flask import Flask, current_app

import receipt_intelligence.settings as settings
from receipt_intelligence.adapters.observability import JsonlEventSink
from receipt_intelligence.adapters.storage.sqlite.model_calls import (
    SQLiteModelCallRepository,
)
from receipt_intelligence.application.ports import EventSink, JobDispatcher, OcrEngine
from receipt_intelligence.application.use_cases.jobs import JobUseCases
from receipt_intelligence.application.use_cases.model_calls import ModelCallUseCases
from receipt_intelligence.application.use_cases.query import AskReceipts, ReceiptQueryExecutor
from receipt_intelligence.application.use_cases.receipts import ReceiptUseCases
from receipt_intelligence.application.use_cases.reviews import ReviewUseCases
from receipt_intelligence.application.use_cases.runtime import RuntimeUseCases
from receipt_intelligence.composition import build_job_dispatcher, build_ocr_engine
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
    job_dispatcher: JobDispatcher
    query_executor: ReceiptQueryExecutor
    model_calls: ModelCallUseCases

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        self.job_dispatcher.shutdown(wait=wait, cancel_futures=cancel_futures)
        close = getattr(self.query_executor, "close", None)
        if callable(close):
            close()


def init_app_services(
    app: Flask,
    *,
    job_store: JobStore | None = None,
    receipt_db: ReceiptDatabase | None = None,
    query_telemetry: EventSink | None = None,
    receipt_query_service: ReceiptQueryExecutor | None = None,
    runtime_paths: RuntimePaths | None = None,
    ocr_engine: OcrEngine | None = None,
    job_dispatcher: JobDispatcher | None = None,
) -> AppServices:
    resolved_paths = runtime_paths or settings.RUNTIME_PATHS
    telemetry_path = (
        resolved_paths.logs_dir / "query_events.jsonl"
        if runtime_paths is not None
        else settings.QUERY_TELEMETRY_PATH
    )
    resolved_store = job_store or JobStore(settings.RESULTS_DIR)
    resolved_database = receipt_db or ReceiptDatabase(settings.RECEIPT_DB_PATH)
    model_call_repository = SQLiteModelCallRepository(
        resolved_database.db_path, enabled=settings.MODEL_CALL_TELEMETRY_ENABLED
    )
    resolved_telemetry = query_telemetry or JsonlEventSink(
        telemetry_path,
        enabled=settings.QUERY_TELEMETRY_ENABLED,
    )
    owns_query_service = receipt_query_service is None
    if receipt_query_service is None:
        from receipt_intelligence.rag_sql.application import (
            build_receipt_query_service_from_settings,
        )

        resolved_query_service = build_receipt_query_service_from_settings(
            telemetry_sink=resolved_telemetry,
            model_call_sink=model_call_repository,
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
    owns_dispatcher = job_dispatcher is None
    resolved_dispatcher = job_dispatcher or build_job_dispatcher(
        resolved_store,
        processor,
        max_workers=settings.JOB_WORKER_MAX_WORKERS,
        queue_capacity=settings.JOB_QUEUE_CAPACITY,
        claim_lease_seconds=settings.JOB_CLAIM_LEASE_SECONDS,
        maintenance_interval_seconds=settings.JOB_MAINTENANCE_INTERVAL_SECONDS,
    )
    services = AppServices(
        jobs=JobUseCases(resolved_store, processor, resolved_dispatcher),
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
        runtime=RuntimeUseCases(RuntimeInformationService(resolved_database, resolved_paths)),
        job_dispatcher=resolved_dispatcher,
        query_executor=resolved_query_service,
        model_calls=ModelCallUseCases(model_call_repository),
    )
    app.extensions[_EXTENSION_KEY] = services
    if settings.JOB_RECOVER_PENDING:
        resolved_dispatcher.recover_pending()
    if owns_dispatcher or owns_query_service:
        atexit.register(services.shutdown, wait=True, cancel_futures=False)
    return services


def get_app_services() -> AppServices:
    services = current_app.extensions.get(_EXTENSION_KEY)
    if not isinstance(services, AppServices):
        raise RuntimeError("Receipt Intelligence application use cases are not initialized.")
    return services


def shutdown_app_services(
    app: Flask,
    *,
    wait: bool = True,
    cancel_futures: bool = False,
) -> None:
    services = app.extensions.get(_EXTENSION_KEY)
    if isinstance(services, AppServices):
        services.shutdown(wait=wait, cancel_futures=cancel_futures)


__all__ = [
    "AppServices",
    "get_app_services",
    "init_app_services",
    "shutdown_app_services",
]
