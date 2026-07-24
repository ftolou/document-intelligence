"""Runtime information assembled from settings, probes, and persistent storage."""

from __future__ import annotations

from typing import Any

import receipt_intelligence.settings as settings
from receipt_intelligence.app_version import app_version_payload, get_app_version
from receipt_intelligence.runtime.readiness import build_readiness_report
from receipt_intelligence.runtime.paths import RuntimePaths
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


class RuntimeInformationService:
    def __init__(self, receipt_db: ReceiptDatabase, runtime_paths: RuntimePaths) -> None:
        self._receipt_db = receipt_db
        self._runtime_paths = runtime_paths

    def health(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": True}
        payload.update(app_version_payload())
        return payload

    def readiness(self) -> dict[str, Any]:
        payload = build_readiness_report(
            database=self._receipt_db,
            runtime_paths=self._runtime_paths,
            ollama_url=settings.OLLAMA_URL,
            vlm_service_url=settings.VLM_SERVICE_URL,
            probe_ollama=settings.READINESS_PROBE_OLLAMA,
            probe_vlm=settings.READINESS_PROBE_VLM and settings.VLM_ENABLED,
            require_ollama=settings.READINESS_REQUIRE_OLLAMA,
            require_vlm=settings.READINESS_REQUIRE_VLM,
            timeout_seconds=settings.READINESS_TIMEOUT_SECONDS,
        )
        payload.update(app_version_payload())
        return payload

    def configuration(self) -> dict[str, Any]:
        return {
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
            "gpu_orchestration": settings.VLM_GPU_ORCHESTRATION,
            "unload_llm_before_vlm": settings.OLLAMA_UNLOAD_BEFORE_VLM,
            "reload_llm_after_vlm": settings.OLLAMA_RELOAD_AFTER_VLM,
            "ollama_control_mode": settings.OLLAMA_CONTROL_MODE,
            "runtime_layout": settings.RUNTIME_LAYOUT,
            "runtime_paths": self._runtime_paths.as_dict(),
            "batch_input_dir": str(settings.BATCH_INPUT_DIR),
            "batch_max_files": settings.BATCH_MAX_FILES,
            "batch_recursive_default": settings.BATCH_RECURSIVE_DEFAULT,
            "version": get_app_version(),
            "app_version": get_app_version(),
            "receipt_db_path": str(self._receipt_db.db_path),
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


__all__ = ["RuntimeInformationService"]
