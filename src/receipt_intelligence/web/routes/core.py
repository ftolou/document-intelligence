"""Root, health, and runtime-configuration endpoints."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify

import receipt_intelligence.settings as settings
from receipt_intelligence.app_version import app_version_payload, get_app_version
from receipt_intelligence.observability.readiness import build_readiness_report
from receipt_intelligence.web.dependencies import get_app_services

core_bp = Blueprint("core", __name__)


@core_bp.get("/")
def index():
    return current_app.send_static_file("index.html")


@core_bp.get("/health")
def health():
    payload = {"ok": True}
    payload.update(app_version_payload())
    return jsonify(payload)


@core_bp.get("/api/readiness")
def readiness():
    services = get_app_services()
    payload = build_readiness_report(
        database=services.receipt_db,
        runtime_paths=services.runtime_paths,
        ollama_url=settings.OLLAMA_URL,
        vlm_service_url=settings.VLM_SERVICE_URL,
        probe_ollama=settings.READINESS_PROBE_OLLAMA,
        probe_vlm=settings.READINESS_PROBE_VLM and settings.VLM_ENABLED,
        require_ollama=settings.READINESS_REQUIRE_OLLAMA,
        require_vlm=settings.READINESS_REQUIRE_VLM,
        timeout_seconds=settings.READINESS_TIMEOUT_SECONDS,
    )
    payload.update(app_version_payload())
    return jsonify(payload), 200 if payload["ready"] else 503


@core_bp.get("/api/config")
def config():
    return jsonify(
        {
            "ollama_url": settings.OLLAMA_URL,
            "model": settings.OLLAMA_MODEL,
            "num_ctx": settings.NUM_CTX,
            "num_predict": settings.NUM_PREDICT,
            "max_lines_for_llm": settings.MAX_LINES_FOR_LLM,
            "ocr_lang": settings.OCR_LANG,
            "ocr_device": settings.OCR_DEVICE,
            "ocr_max_side_limit": settings.OCR_MAX_SIDE_LIMIT,
            "json_retry_count": settings.LLM_JSON_RETRY_COUNT,
            "format_json": settings.OLLAMA_FORMAT_JSON,
            "vlm_enabled": settings.VLM_ENABLED,
            "vlm_backend": settings.VLM_BACKEND,
            "vlm_service_url": settings.VLM_SERVICE_URL,
            "vlm_timeout_seconds": settings.VLM_TIMEOUT_SECONDS,
            "vlm_correction_enabled": settings.VLM_CORRECTION_ENABLED,
            "vlm_gpu_orchestration": settings.VLM_GPU_ORCHESTRATION,
            "ollama_unload_before_vlm": settings.OLLAMA_UNLOAD_BEFORE_VLM,
            "ollama_reload_after_vlm": settings.OLLAMA_RELOAD_AFTER_VLM,
            "ollama_control_mode": settings.OLLAMA_CONTROL_MODE,
            "runtime_layout": settings.RUNTIME_LAYOUT,
            "runtime_paths": settings.RUNTIME_PATHS.as_dict(),
            "batch_input_dir": str(settings.BATCH_INPUT_DIR),
            "batch_max_files": settings.BATCH_MAX_FILES,
            "batch_recursive_default": settings.BATCH_RECURSIVE_DEFAULT,
            "version": get_app_version(),
            "app_version": get_app_version(),
            "receipt_db_path": str(settings.RECEIPT_DB_PATH),
            "categorization_enabled": settings.CATEGORIZATION_ENABLED,
            "categorization_model": settings.CATEGORIZATION_MODEL,
            "categorization_num_ctx": settings.CATEGORIZATION_NUM_CTX,
            "categorization_num_predict": settings.CATEGORIZATION_NUM_PREDICT,
            "categorization_timeout_seconds": settings.CATEGORIZATION_TIMEOUT_SECONDS,
            "categorization_format_json": settings.CATEGORIZATION_FORMAT_JSON,
            "query_telemetry_enabled": settings.QUERY_TELEMETRY_ENABLED,
            "query_telemetry_path": str(settings.QUERY_TELEMETRY_PATH),
            "query_engine": {
                "name": "rag_sql",
                "orchestrator": "langgraph",
                "graph_version": "rag_sql_graph_v2",
            },
            "readiness_probe_ollama": settings.READINESS_PROBE_OLLAMA,
            "readiness_probe_vlm": settings.READINESS_PROBE_VLM,
        }
    )
