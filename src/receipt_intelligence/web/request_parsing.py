"""HTTP form parsing for receipt-processing options."""

from __future__ import annotations

from typing import Any

from flask import request

import receipt_intelligence.settings as settings


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in settings.ALLOWED_EXTENSIONS


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def form_int(name: str, default: int) -> int:
    try:
        return int(request.form.get(name, default))
    except Exception:
        return default


def form_float(name: str, default: float) -> float:
    try:
        return float(request.form.get(name, default))
    except Exception:
        return default


def build_options_from_request() -> dict[str, Any]:
    """Read parser/runtime options from the current Flask request form."""
    return {
        "ollama_url": settings.OLLAMA_URL,
        "model": request.form.get("model") or settings.OLLAMA_MODEL,
        "num_ctx": form_int("num_ctx", settings.NUM_CTX),
        "num_predict": form_int("num_predict", settings.NUM_PREDICT),
        "keep_alive": (
            request.form.get("keep_alive")
            if request.form.get("keep_alive") is not None
            else settings.OLLAMA_KEEP_ALIVE
        ),
        "llm_timeout_seconds": form_float("llm_timeout_seconds", settings.LLM_TIMEOUT_SECONDS),
        "ocr_lang": request.form.get("ocr_lang") or settings.OCR_LANG,
        "ocr_device": request.form.get("ocr_device") or settings.OCR_DEVICE,
        "max_crops": form_int("max_crops", settings.EXTRACTION_MAX_CROPS),
        "validation_tolerance": form_float("validation_tolerance", settings.VALIDATION_TOLERANCE),
        "json_retry_count": form_int("json_retry_count", settings.LLM_JSON_RETRY_COUNT),
        "format_json": as_bool(request.form.get("format_json"), settings.OLLAMA_FORMAT_JSON),
        "correction_enabled": as_bool(
            request.form.get("correction_enabled"),
            settings.CORRECTION_ENABLED,
        ),
        "categorization_enabled": as_bool(
            request.form.get("categorization_enabled"),
            settings.CATEGORIZATION_ENABLED,
        ),
        "categorization_model": (
            request.form.get("categorization_model") or settings.CATEGORIZATION_MODEL
        ),
        "categorization_num_ctx": form_int(
            "categorization_num_ctx", settings.CATEGORIZATION_NUM_CTX
        ),
        "categorization_num_predict": form_int(
            "categorization_num_predict", settings.CATEGORIZATION_NUM_PREDICT
        ),
        "categorization_timeout_seconds": form_float(
            "categorization_timeout_seconds",
            settings.CATEGORIZATION_TIMEOUT_SECONDS,
        ),
        "categorization_format_json": as_bool(
            request.form.get("categorization_format_json"),
            settings.CATEGORIZATION_FORMAT_JSON,
        ),
    }
