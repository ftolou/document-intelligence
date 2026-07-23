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


def normalize_vlm_backend(backend: str, service_url: str) -> str:
    """Route retained local-backend form values to the separate VLM service."""
    value = (backend or "").strip()
    if not value:
        return settings.VLM_BACKEND
    local_names = {"paddleocr_vl", "paddleocr-vl", "paddleocrvl", "local"}
    if value.lower() in local_names and not getattr(settings, "VLM_ALLOW_LOCAL_BACKEND", False):
        return "http_service" if service_url else value
    return value


def build_options_from_request() -> dict[str, Any]:
    """Read parser/runtime options from the current Flask request form."""
    service_url = request.form.get("vlm_service_url") or settings.VLM_SERVICE_URL
    return {
        "ollama_url": request.form.get("ollama_url") or settings.OLLAMA_URL,
        "model": request.form.get("model") or settings.OLLAMA_MODEL,
        "num_ctx": form_int("num_ctx", settings.NUM_CTX),
        "num_predict": form_int("num_predict", settings.NUM_PREDICT),
        "keep_alive": (
            request.form.get("keep_alive")
            if request.form.get("keep_alive") is not None
            else settings.OLLAMA_KEEP_ALIVE
        ),
        "llm_timeout_seconds": form_float("llm_timeout_seconds", settings.LLM_TIMEOUT_SECONDS),
        "max_lines_for_llm": form_int("max_lines_for_llm", settings.MAX_LINES_FOR_LLM),
        "ocr_lang": request.form.get("ocr_lang") or settings.OCR_LANG,
        "ocr_device": request.form.get("ocr_device") or settings.OCR_DEVICE,
        "ocr_max_side_limit": form_int("ocr_max_side_limit", settings.OCR_MAX_SIDE_LIMIT),
        "ocr_use_angle_cls": as_bool(
            request.form.get("ocr_use_angle_cls"), settings.OCR_USE_ANGLE_CLS
        ),
        "ocr_det_limit_side_len": form_int(
            "ocr_det_limit_side_len", settings.OCR_DET_LIMIT_SIDE_LEN
        ),
        "validation_tolerance": form_float("validation_tolerance", settings.VALIDATION_TOLERANCE),
        "json_retry_count": form_int("json_retry_count", settings.LLM_JSON_RETRY_COUNT),
        "format_json": as_bool(request.form.get("format_json"), settings.OLLAMA_FORMAT_JSON),
        "vlm_enabled": as_bool(request.form.get("vlm_enabled"), settings.VLM_ENABLED),
        "vlm_service_url": service_url,
        "vlm_backend": normalize_vlm_backend(
            request.form.get("vlm_backend") or settings.VLM_BACKEND,
            service_url,
        ),
        "vlm_command": (
            request.form.get("vlm_command")
            if request.form.get("vlm_command") is not None
            else settings.VLM_COMMAND
        ),
        "vlm_timeout_seconds": form_float("vlm_timeout_seconds", settings.VLM_TIMEOUT_SECONDS),
        "vlm_max_chars": form_int("vlm_max_chars", settings.VLM_MAX_CHARS),
        "vlm_correction_enabled": as_bool(
            request.form.get("vlm_correction_enabled"),
            settings.VLM_CORRECTION_ENABLED,
        ),
        "vlm_gpu_orchestration": (
            request.form.get("vlm_gpu_orchestration") or settings.VLM_GPU_ORCHESTRATION
        ),
        "ollama_unload_before_vlm": as_bool(
            request.form.get("ollama_unload_before_vlm"),
            settings.OLLAMA_UNLOAD_BEFORE_VLM,
        ),
        "ollama_reload_after_vlm": as_bool(
            request.form.get("ollama_reload_after_vlm"),
            settings.OLLAMA_RELOAD_AFTER_VLM,
        ),
        "ollama_control_mode": (
            request.form.get("ollama_control_mode") or settings.OLLAMA_CONTROL_MODE
        ),
        "ollama_control_timeout_seconds": form_float(
            "ollama_control_timeout_seconds",
            settings.OLLAMA_CONTROL_TIMEOUT_SECONDS,
        ),
        "ollama_unload_command": (
            request.form.get("ollama_unload_command")
            if request.form.get("ollama_unload_command") is not None
            else settings.OLLAMA_UNLOAD_COMMAND
        ),
        "ollama_start_command": (
            request.form.get("ollama_start_command")
            if request.form.get("ollama_start_command") is not None
            else settings.OLLAMA_START_COMMAND
        ),
        "ollama_reload_prompt": (
            request.form.get("ollama_reload_prompt")
            if request.form.get("ollama_reload_prompt") is not None
            else settings.OLLAMA_RELOAD_PROMPT
        ),
        "ollama_gpu_handoff_wait_seconds": form_float(
            "ollama_gpu_handoff_wait_seconds",
            settings.OLLAMA_GPU_HANDOFF_WAIT_SECONDS,
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
