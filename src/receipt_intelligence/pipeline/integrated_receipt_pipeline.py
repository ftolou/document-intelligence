#!/usr/bin/env python3
"""Canonical typed receipt extraction entry point and deprecated keyword wrapper."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from receipt_intelligence.extraction.composition import build_extraction_dependencies
from receipt_intelligence.extraction.config import ExtractionRequest
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.dependencies import ExtractionDependencies
from receipt_intelligence.extraction.factory import build_extraction_workflow
from receipt_intelligence.observability.timing import utc_now_iso


def run_receipt_extraction(
    request: ExtractionRequest,
    *,
    dependencies: ExtractionDependencies | None = None,
) -> dict[str, Any]:
    """Run one receipt image through the selected extraction backend."""

    if request.extraction_backend == "openai_one_shot":
        from receipt_intelligence.extraction.openai_observability import (
            build_observed_openai_client,
            publish_openai_extraction_metrics,
        )
        from receipt_intelligence.extraction.openai_one_shot import (
            run_openai_one_shot_extraction,
        )

        started_at = utc_now_iso()
        started = time.perf_counter()
        try:
            client = build_observed_openai_client(request)
            result = run_openai_one_shot_extraction(request, client=client)
        except Exception as exc:
            publish_openai_extraction_metrics(
                request,
                status="failed",
                started_at=started_at,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                stages=(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        observability = (
            result.get("observability") if isinstance(result.get("observability"), dict) else {}
        )
        stage_trace = observability.get("stage_trace")
        stages = tuple(value for value in stage_trace if isinstance(value, dict)) if isinstance(stage_trace, list) else ()
        metrics_path = publish_openai_extraction_metrics(
            request,
            status="completed",
            started_at=started_at,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            stages=stages,
        )
        paths = dict(result.get("paths") or {})
        paths["extraction_metrics"] = metrics_path
        result["paths"] = paths
        result["observability"] = {
            **observability,
            "stage_trace": [dict(value) for value in stages],
            "metrics_path": metrics_path,
        }
        return result

    context = ExtractionContext(
        config=request,
        dependencies=dependencies or build_extraction_dependencies(request),
    )
    completed = build_extraction_workflow().run(context)
    return _application_result(completed)


def _application_result(context: ExtractionContext) -> dict[str, Any]:
    finalized = context.require_finalized().result
    if finalized is None:
        raise RuntimeError("Extraction workflow finished without a finalization result.")
    result = finalized.as_application_result()
    result["paths"] = dict(context.paths)
    result["logs"] = list(context.logs)
    result["observability"] = {
        "stage_trace": [dict(entry) for entry in context.stage_trace],
        "metrics_path": context.paths.get("extraction_metrics"),
    }
    return result


def run_integrated_receipt_pipeline(
    *,
    result_dir: Path,
    run_id: str,
    ollama_url: str,
    model: str,
    source_image_path: Path | None = None,
    ocr_json_path: Path | None = None,
    tolerance: float = 0.03,
    max_reocr_images: int = 4,
    repair_ocr_min_score: float = 0.20,
    ocr_lang: str = "german",
    ocr_device: str = "cpu",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    spatial_canvas_width: int = 112,
    max_lines_for_llm: int = 220,
    num_ctx: int = 16384,
    num_predict: int = 8192,
    keep_alive: str | None = None,
    llm_timeout_seconds: float = 300.0,
    json_retry_count: int = 1,
    format_json: bool = True,
    vlm_backend: str = "http_service",
    vlm_service_url: str = "http://receipt-vlm:7870",
    vlm_timeout_seconds: float = 900.0,
    vlm_max_chars: int = 12000,
    correction_enabled: bool = True,
    categorization_enabled: bool = True,
    categorization_model: str | None = None,
    categorization_num_ctx: int = 16384,
    categorization_num_predict: int = 4096,
    categorization_timeout_seconds: float = 180.0,
    categorization_format_json: bool = True,
    gpu_orchestration: str | None = None,
    unload_llm_before_vlm: bool | None = None,
    reload_llm_after_vlm: bool | None = None,
    ollama_control_mode: str = "api",
    ollama_control_timeout_seconds: float = 120.0,
    ollama_unload_command: str = "",
    ollama_start_command: str = "",
    ollama_reload_prompt: str = "ok",
    ollama_gpu_handoff_wait_seconds: float = 0.0,
    **legacy_options: Any,
) -> dict[str, Any]:
    """Deprecated keyword adapter that always runs the canonical pipeline.

    Historical OCR/VLM-only options remain in the signature for one compatibility
    window but no longer affect extraction.
    """

    del (
        ocr_json_path,
        repair_ocr_min_score,
        spatial_canvas_width,
        max_lines_for_llm,
        vlm_backend,
        vlm_service_url,
        vlm_timeout_seconds,
        vlm_max_chars,
        gpu_orchestration,
        unload_llm_before_vlm,
        reload_llm_after_vlm,
        ollama_control_mode,
        ollama_control_timeout_seconds,
        ollama_unload_command,
        ollama_start_command,
        ollama_reload_prompt,
        ollama_gpu_handoff_wait_seconds,
    )
    if legacy_options:
        names = ", ".join(sorted(repr(name) for name in legacy_options))
        raise TypeError(f"Unsupported extraction option(s): {names}")
    if source_image_path is None:
        raise TypeError("source_image_path is required by the image-first extraction pipeline")
    warnings.warn(
        "run_integrated_receipt_pipeline() is deprecated; construct ExtractionRequest and "
        "call run_receipt_extraction() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    request = ExtractionRequest(
        source_image_path=source_image_path,
        result_dir=result_dir,
        run_id=run_id,
        ollama_url=ollama_url,
        model=model,
        tolerance=tolerance,
        max_crops=max_reocr_images or 4,
        ocr_lang=ocr_lang,
        ocr_device=ocr_device,
        progress_callback=progress_callback,
        num_ctx=num_ctx,
        num_predict=num_predict,
        keep_alive=keep_alive,
        llm_timeout_seconds=llm_timeout_seconds,
        json_retry_count=json_retry_count,
        format_json=format_json,
        correction_enabled=correction_enabled,
        categorization_enabled=categorization_enabled,
        categorization_model=categorization_model,
        categorization_num_ctx=categorization_num_ctx,
        categorization_num_predict=categorization_num_predict,
        categorization_timeout_seconds=categorization_timeout_seconds,
        categorization_format_json=categorization_format_json,
    )
    return run_receipt_extraction(request)


__all__ = ["run_receipt_extraction", "run_integrated_receipt_pipeline"]
