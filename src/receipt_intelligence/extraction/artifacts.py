"""Artifact naming and persistence helpers for extraction stages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def copy_alias(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copyfile(source, destination)


def build_artifact_paths(result_dir: Path, run_id: str) -> dict[str, Path]:
    """Return stable artifact paths used by the existing UI and scripts."""

    return {
        "ocr_context": result_dir / f"{run_id}_v14_ocr_context.json",
        "llm_main_prompt": result_dir / f"{run_id}_v14_llm_main_prompt.txt",
        "llm_main_raw": result_dir / f"{run_id}_v14_llm_main_raw.txt",
        "receipt_llm_main": result_dir / f"{run_id}_v14_receipt_llm_main.json",
        "receipt_llm_main_postprocessed": result_dir
        / f"{run_id}_v14_receipt_llm_main_postprocessed.json",
        "consistency_postprocess": result_dir / f"{run_id}_v14_consistency_postprocess.json",
        "layout_context": result_dir / f"{run_id}_v14_6_layout_context.json",
        "compact_evidence": result_dir / f"{run_id}_v14_6_compact_evidence.json",
        "compact_evidence_text": result_dir / f"{run_id}_v14_6_compact_evidence.txt",
        "grouped_evidence": result_dir / f"{run_id}_v14_6_grouped_evidence.json",
        "validation_report": result_dir / f"{run_id}_v14_validation_report.json",
        "semantic_suspicion": result_dir / f"{run_id}_semantic_suspicion.json",
        "semantic_suspicion_patch_corrected": result_dir
        / f"{run_id}_semantic_suspicion_patch_corrected.json",
        "vlm_raw_output": result_dir / f"{run_id}_v14_6_vlm_raw_output.json",
        "visual_evidence": result_dir / f"{run_id}_v14_6_visual_evidence.json",
        "visual_evidence_text": result_dir / f"{run_id}_v14_6_visual_evidence.txt",
        "table_arbitration": result_dir / f"{run_id}_v14_18_table_arbitration.json",
        "spatial_document_map": result_dir / f"{run_id}_spatial_document_map.json",
        "spatial_canvas": result_dir / f"{run_id}_spatial_canvas.txt",
        "spatial_overview": result_dir / f"{run_id}_spatial_overview.json",
        "region_reocr": result_dir / f"{run_id}_v14_13_region_reocr.json",
        "right_column_reocr": result_dir / f"{run_id}_v14_6_right_column_reocr.json",
        "line_price_fusion": result_dir / f"{run_id}_line_price_fusion.json",
        "receipt_line_price_fused": result_dir / f"{run_id}_receipt_line_price_fused.json",
        "validation_report_line_price_fused": result_dir
        / f"{run_id}_validation_report_line_price_fused.json",
        "correction_patch_prompt": result_dir / f"{run_id}_v14_18_correction_patch_prompt.txt",
        "correction_patch_raw": result_dir / f"{run_id}_v14_18_correction_patch_raw.txt",
        "correction_patch_result": result_dir / f"{run_id}_v14_18_correction_patch_result.json",
        "receipt_patch_corrected": result_dir / f"{run_id}_v14_18_receipt_patch_corrected.json",
        "validation_report_patch_corrected": result_dir
        / f"{run_id}_v14_18_validation_report_patch_corrected.json",
        "receipt_final": result_dir / f"{run_id}_receipt_final.json",
        "receipt_final_reconciled": result_dir / f"{run_id}_receipt_final_reconciled.json",
        "receipt_final_categorized": result_dir / f"{run_id}_receipt_final_categorized.json",
        "categorization_prompt": result_dir / f"{run_id}_v14_14_categorization_prompt.txt",
        "categorization_raw": result_dir / f"{run_id}_v14_14_categorization_raw.txt",
        "categorization_result": result_dir / f"{run_id}_v14_14_categorization_result.json",
        "reconciliation_report": result_dir / f"{run_id}_reconciliation_report.json",
        "pipeline_meta": result_dir / f"{run_id}_pipeline_meta.json",
        "stage_trace": result_dir / f"{run_id}_extraction_stage_trace.json",
        "extraction_metrics": result_dir / f"{run_id}_extraction_metrics.json",
        "gpu_orchestration_before_vlm": result_dir / f"{run_id}_gpu_orchestration_before_vlm.json",
        "gpu_orchestration_after_vlm": result_dir / f"{run_id}_gpu_orchestration_after_vlm.json",
    }


def publish_latest_aliases(paths: dict[str, Path], result_dir: Path) -> None:
    aliases = {
        "latest_v14_ocr_context": ("ocr_context", "latest_v14_ocr_context.json"),
        "latest_v14_llm_main_prompt": ("llm_main_prompt", "latest_v14_llm_main_prompt.txt"),
        "latest_v14_llm_main_raw": ("llm_main_raw", "latest_v14_llm_main_raw.txt"),
        "latest_v14_receipt_llm_main": ("receipt_llm_main", "latest_v14_receipt_llm_main.json"),
        "latest_v14_6_layout_context": ("layout_context", "latest_v14_6_layout_context.json"),
        "latest_v14_6_compact_evidence": ("compact_evidence", "latest_v14_6_compact_evidence.json"),
        "latest_v14_6_compact_evidence_text": (
            "compact_evidence_text",
            "latest_v14_6_compact_evidence.txt",
        ),
        "latest_v14_6_grouped_evidence": ("grouped_evidence", "latest_v14_6_grouped_evidence.json"),
        "latest_v14_6_vlm_raw_output": ("vlm_raw_output", "latest_v14_6_vlm_raw_output.json"),
        "latest_v14_6_right_column_reocr": (
            "right_column_reocr",
            "latest_v14_6_right_column_reocr.json",
        ),
        "latest_v14_6_visual_evidence": ("visual_evidence", "latest_v14_6_visual_evidence.json"),
        "latest_v14_6_visual_evidence_text": (
            "visual_evidence_text",
            "latest_v14_6_visual_evidence.txt",
        ),
        "latest_line_price_fusion": (
            "line_price_fusion",
            "latest_line_price_fusion.json",
        ),
        "latest_receipt_line_price_fused": (
            "receipt_line_price_fused",
            "latest_receipt_line_price_fused.json",
        ),
        "latest_validation_report_line_price_fused": (
            "validation_report_line_price_fused",
            "latest_validation_report_line_price_fused.json",
        ),
        "latest_v14_18_table_arbitration": (
            "table_arbitration",
            "latest_v14_18_table_arbitration.json",
        ),
        "latest_spatial_document_map": (
            "spatial_document_map",
            "latest_spatial_document_map.json",
        ),
        "latest_spatial_canvas": ("spatial_canvas", "latest_spatial_canvas.txt"),
        "latest_spatial_overview": (
            "spatial_overview",
            "latest_spatial_overview.json",
        ),
        "latest_v14_13_region_reocr": ("region_reocr", "latest_v14_13_region_reocr.json"),
        "latest_v14_18_correction_patch_prompt": (
            "correction_patch_prompt",
            "latest_v14_18_correction_patch_prompt.txt",
        ),
        "latest_v14_18_correction_patch_raw": (
            "correction_patch_raw",
            "latest_v14_18_correction_patch_raw.txt",
        ),
        "latest_v14_18_correction_patch_result": (
            "correction_patch_result",
            "latest_v14_18_correction_patch_result.json",
        ),
        "latest_v14_18_receipt_patch_corrected": (
            "receipt_patch_corrected",
            "latest_v14_18_receipt_patch_corrected.json",
        ),
        "latest_v14_18_validation_report_patch_corrected": (
            "validation_report_patch_corrected",
            "latest_v14_18_validation_report_patch_corrected.json",
        ),
        "latest_v14_validation_report": ("validation_report", "latest_v14_validation_report.json"),
        "latest_semantic_suspicion": (
            "semantic_suspicion",
            "latest_semantic_suspicion.json",
        ),
        "latest_semantic_suspicion_patch_corrected": (
            "semantic_suspicion_patch_corrected",
            "latest_semantic_suspicion_patch_corrected.json",
        ),
        "latest_receipt_final": ("receipt_final", "latest_receipt_final.json"),
        "latest_receipt_final_reconciled": (
            "receipt_final_reconciled",
            "latest_receipt_final_reconciled.json",
        ),
        "latest_receipt_final_categorized": (
            "receipt_final_categorized",
            "latest_receipt_final_categorized.json",
        ),
        "latest_v14_14_categorization_prompt": (
            "categorization_prompt",
            "latest_v14_14_categorization_prompt.txt",
        ),
        "latest_v14_14_categorization_raw": (
            "categorization_raw",
            "latest_v14_14_categorization_raw.txt",
        ),
        "latest_v14_14_categorization_result": (
            "categorization_result",
            "latest_v14_14_categorization_result.json",
        ),
        "latest_reconciliation_report": (
            "reconciliation_report",
            "latest_reconciliation_report.json",
        ),
        "latest_pipeline_meta": ("pipeline_meta", "latest_pipeline_meta.json"),
        "latest_extraction_stage_trace": ("stage_trace", "latest_extraction_stage_trace.json"),
        "latest_extraction_metrics": ("extraction_metrics", "latest_extraction_metrics.json"),
    }
    for alias_key, (source_key, filename) in aliases.items():
        source = paths.get(source_key)
        if source is None or not source.exists():
            continue
        alias = result_dir / filename
        copy_alias(source, alias)
        paths[alias_key] = alias
