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
) -> dict[str, Any]:
    """Run receipt extraction from an explicit immutable request contract."""

    context = ExtractionContext(
        config=request,
        dependencies=dependencies or build_extraction_dependencies(request),
    )
    workflow = build_default_extraction_workflow()
    return workflow.run(context).as_result()


def run_integrated_receipt_pipeline(
    *,
    ocr_json_path: Path,
    result_dir: Path,
    run_id: str,
    ollama_url: str,
    model: str,
    tolerance: float = 0.03,
    skip_row_llm: bool = False,
    active_line_repair: bool = False,
    max_repair_passes: int = 0,
    max_repair_rois: int = 0,
    max_repair_variants: int = 0,
    max_reocr_images: int = 0,
    repair_time_budget_seconds: float = 0.0,
    repair_ocr_min_score: float = 0.20,
    ocr_lang: str = "german",
    ocr_device: str = "cpu",
    ocr_det_model: str | None = None,
    ocr_rec_model: str | None = "latin_PP-OCRv5_mobile_rec",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    extraction_strategy: str = "current",
    spatial_overview_num_ctx: int = 16384,
    spatial_overview_num_predict: int = 4096,
    spatial_overview_timeout_seconds: float = 180.0,
    spatial_canvas_width: int = 112,
    max_lines_for_llm: int = 260,
    num_ctx: int = 24384,
    num_predict: int = 8192,
    keep_alive: str | None = None,
    llm_timeout_seconds: float = 240.0,
    json_retry_count: int = 1,
    format_json: bool = True,
    source_image_path: Path | None = None,
    vlm_enabled: bool = False,
    vlm_backend: str = "http_service",
    vlm_service_url: str = "http://receipt-vlm:7870",
    vlm_command: str = "",
    vlm_timeout_seconds: float = 180.0,
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
        "skip_row_llm": skip_row_llm,
        "active_line_repair": active_line_repair,
        "max_repair_passes": max_repair_passes,
        "max_repair_rois": max_repair_rois,
        "max_repair_variants": max_repair_variants,
        "max_reocr_images": max_reocr_images,
        "repair_time_budget_seconds": repair_time_budget_seconds,
        "repair_ocr_min_score": repair_ocr_min_score,
        "ocr_lang": ocr_lang,
        "ocr_device": ocr_device,
        "ocr_det_model": ocr_det_model,
        "ocr_rec_model": ocr_rec_model,
        "progress_callback": progress_callback,
        "extraction_strategy": extraction_strategy,
        "spatial_overview_num_ctx": spatial_overview_num_ctx,
        "spatial_overview_num_predict": spatial_overview_num_predict,
        "spatial_overview_timeout_seconds": spatial_overview_timeout_seconds,
        "spatial_canvas_width": spatial_canvas_width,
        "max_lines_for_llm": max_lines_for_llm,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "keep_alive": keep_alive,
        "llm_timeout_seconds": llm_timeout_seconds,
        "json_retry_count": json_retry_count,
        "format_json": format_json,
        "source_image_path": source_image_path,
        "vlm_enabled": vlm_enabled,
        "vlm_backend": vlm_backend,
        "vlm_service_url": vlm_service_url,
        "vlm_command": vlm_command,
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
    return run_receipt_extraction(request)


__all__ = [
    "run_receipt_extraction",
    "run_integrated_receipt_pipeline",
    "_report_score",
    "_should_run_visual_layer",
    "_merge_visual_evidence",
]
