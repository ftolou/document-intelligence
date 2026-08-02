#!/usr/bin/env python3
"""Typed receipt extraction entry point plus a strict compatibility adapter."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from receipt_intelligence.composition import build_extraction_dependencies
from receipt_intelligence.extraction import (
    ExtractionContext,
    ExtractionRequest,
    build_default_extraction_workflow,
)
from receipt_intelligence.extraction.compatibility import extraction_request_from_mapping
from receipt_intelligence.extraction.dependencies import ExtractionDependencies
from receipt_intelligence.extraction.strategy import (
    ExtractionStrategy,
    resolve_extraction_strategy,
)
from receipt_intelligence.extraction.support import (
    merge_visual_evidence as _merge_visual_evidence,
)
from receipt_intelligence.extraction.support import report_score as _report_score
from receipt_intelligence.extraction.support import (
    should_run_visual_layer as _should_run_visual_layer,
)


def run_receipt_extraction(
    request: ExtractionRequest,
    *,
    dependencies: ExtractionDependencies | None = None,
    extraction_strategy: str | ExtractionStrategy | None = None,
) -> dict[str, Any]:
    """Run one receipt through the selected extraction workflow.

    ``current`` remains the default and preserves the existing production path.
    ``next`` activates the staged Paddle/Qwen/Gemma workflow introduced in Phases 1-6.
    """

    strategy = resolve_extraction_strategy(extraction_strategy)
    if strategy is ExtractionStrategy.NEXT:
        from receipt_intelligence.extraction.next_composition import (
            build_next_extraction_dependencies,
        )
        from receipt_intelligence.extraction.next_factory import (
            build_next_extraction_workflow,
        )

        context = ExtractionContext(
            config=request,
            dependencies=dependencies or build_next_extraction_dependencies(request),
        )
        completed = build_next_extraction_workflow().run(context)
        return _next_application_result(completed)

    context = ExtractionContext(
        config=request,
        dependencies=dependencies or build_extraction_dependencies(request),
    )
    workflow = build_default_extraction_workflow()
    return workflow.run(context).as_result()


def _next_application_result(context: ExtractionContext) -> dict[str, Any]:
    finalized = context.require_finalized().next_finalization
    if finalized is None:
        raise RuntimeError("Next extraction workflow finished without finalization result.")
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
    ocr_json_path: Path,
    result_dir: Path,
    run_id: str,
    ollama_url: str,
    model: str,
    tolerance: float = 0.03,
    max_reocr_images: int = 0,
    repair_ocr_min_score: float = 0.20,
    ocr_lang: str = "german",
    ocr_device: str = "cpu",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    spatial_canvas_width: int = 112,
    max_lines_for_llm: int = 220,
    num_ctx: int = 24384,
    num_predict: int = 8192,
    keep_alive: str | None = None,
    llm_timeout_seconds: float = 300.0,
    json_retry_count: int = 1,
    format_json: bool = True,
    source_image_path: Path | None = None,
    vlm_backend: str = "http_service",
    vlm_service_url: str = "http://receipt-vlm:7870",
    vlm_timeout_seconds: float = 900.0,
    vlm_max_chars: int = 12000,
    correction_enabled: bool = True,
    categorization_enabled: bool = True,
    categorization_model: str | None = None,
    categorization_num_ctx: int = 8192,
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
    extraction_strategy: str | ExtractionStrategy | None = None,
    **legacy_options: Any,
) -> dict[str, Any]:
    """Backward-compatible adapter that rejects unsupported options.

    New application code should construct :class:`ExtractionRequest` and call
    :func:`run_receipt_extraction`. This adapter exists for scripts and external
    callers that still use the historical keyword interface.
    """

    values: dict[str, Any] = {
        "ocr_json_path": ocr_json_path,
        "result_dir": result_dir,
        "run_id": run_id,
        "ollama_url": ollama_url,
        "model": model,
        "tolerance": tolerance,
        "max_reocr_images": max_reocr_images,
        "repair_ocr_min_score": repair_ocr_min_score,
        "ocr_lang": ocr_lang,
        "ocr_device": ocr_device,
        "progress_callback": progress_callback,
        "spatial_canvas_width": spatial_canvas_width,
        "max_lines_for_llm": max_lines_for_llm,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "keep_alive": keep_alive,
        "llm_timeout_seconds": llm_timeout_seconds,
        "json_retry_count": json_retry_count,
        "format_json": format_json,
        "source_image_path": source_image_path,
        "vlm_backend": vlm_backend,
        "vlm_service_url": vlm_service_url,
        "vlm_timeout_seconds": vlm_timeout_seconds,
        "vlm_max_chars": vlm_max_chars,
        "correction_enabled": correction_enabled,
        "categorization_enabled": categorization_enabled,
        "categorization_model": categorization_model,
        "categorization_num_ctx": categorization_num_ctx,
        "categorization_num_predict": categorization_num_predict,
        "categorization_timeout_seconds": categorization_timeout_seconds,
        "categorization_format_json": categorization_format_json,
        "ollama_control_mode": ollama_control_mode,
        "ollama_control_timeout_seconds": ollama_control_timeout_seconds,
        "ollama_unload_command": ollama_unload_command,
        "ollama_start_command": ollama_start_command,
        "ollama_reload_prompt": ollama_reload_prompt,
        "ollama_gpu_handoff_wait_seconds": ollama_gpu_handoff_wait_seconds,
    }
    if gpu_orchestration is not None:
        values["gpu_orchestration"] = gpu_orchestration
    if unload_llm_before_vlm is not None:
        values["unload_llm_before_vlm"] = unload_llm_before_vlm
    if reload_llm_after_vlm is not None:
        values["reload_llm_after_vlm"] = reload_llm_after_vlm
    values.update(legacy_options)

    request = extraction_request_from_mapping(values)
    return run_receipt_extraction(
        request,
        extraction_strategy=extraction_strategy,
    )


__all__ = [
    "run_receipt_extraction",
    "run_integrated_receipt_pipeline",
    "_report_score",
    "_should_run_visual_layer",
    "_merge_visual_evidence",
]
