"""Application-owned dependencies exposed to Flask blueprints."""

from __future__ import annotations

from dataclasses import dataclass

from flask import Flask, current_app

import receipt_intelligence.settings as settings
from receipt_intelligence.application.ports import OcrEngine
from receipt_intelligence.composition import build_ocr_engine
from receipt_intelligence.observability.query import QueryTelemetrySink
from receipt_intelligence.rag_sql.application import (
    ReceiptQueryService,
    build_receipt_query_service_from_settings,
)
from receipt_intelligence.runtime.paths import RuntimePaths
from receipt_intelligence.storage.job_store import JobStore
from receipt_intelligence.storage.receipt_db import ReceiptDatabase

_EXTENSION_KEY = "receipt_intelligence.services"


@dataclass(frozen=True)
class AppServices:
    job_store: JobStore
    receipt_db: ReceiptDatabase
    query_telemetry: QueryTelemetrySink
    receipt_query_service: ReceiptQueryService
    runtime_paths: RuntimePaths
    ocr_engine: OcrEngine


def init_app_services(
    app: Flask,
    *,
    job_store: JobStore | None = None,
    receipt_db: ReceiptDatabase | None = None,
    query_telemetry: QueryTelemetrySink | None = None,
    receipt_query_service: ReceiptQueryService | None = None,
    runtime_paths: RuntimePaths | None = None,
    ocr_engine: OcrEngine | None = None,
) -> AppServices:
    resolved_paths = runtime_paths or settings.RUNTIME_PATHS
    telemetry_path = (
        resolved_paths.logs_dir / "query_events.jsonl"
        if runtime_paths is not None
        else settings.QUERY_TELEMETRY_PATH
    )
    resolved_database = receipt_db or ReceiptDatabase(settings.RECEIPT_DB_PATH)
    resolved_telemetry = query_telemetry or QueryTelemetrySink.from_path(
        telemetry_path,
        enabled=settings.QUERY_TELEMETRY_ENABLED,
    )
    resolved_query_service = receipt_query_service or build_receipt_query_service_from_settings(
        telemetry_sink=resolved_telemetry,
    )
    services = AppServices(
        job_store=job_store or JobStore(settings.RESULTS_DIR),
        receipt_db=resolved_database,
        query_telemetry=resolved_telemetry,
        receipt_query_service=resolved_query_service,
        runtime_paths=resolved_paths,
        ocr_engine=ocr_engine or build_ocr_engine(),
    )
    app.extensions[_EXTENSION_KEY] = services
    return services


def get_app_services() -> AppServices:
    services = current_app.extensions.get(_EXTENSION_KEY)
    if not isinstance(services, AppServices):
        raise RuntimeError("Receipt Intelligence application services are not initialized.")
    return services
