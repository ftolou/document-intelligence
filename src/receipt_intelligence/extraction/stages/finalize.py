"""Finalize validated receipt artifacts, categorization and metadata."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.extraction.artifacts import (
    publish_latest_aliases,
    save_json,
)
from receipt_intelligence.extraction.categorization.items import (
    categorize_receipt_items_llm,
    write_categorization_artifacts,
)
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.state import ExtractionPhase
from receipt_intelligence.extraction.validation.consistency import (
    sanitize_model_warnings,
)


class FinalizationStage:
    name = "finalize"
    input_phase = ExtractionPhase.REPAIRED
    output_phase = ExtractionPhase.FINALIZED

    def run(self, context: ExtractionContext) -> ExtractionContext:
        context.begin_finalization_stage()
        config = context.config
        paths = context.paths
        final_receipt = context.final_receipt
        final_report = context.final_report

        final_receipt, warning_actions = sanitize_model_warnings(final_receipt, final_report)
        context.final_receipt = final_receipt
        if warning_actions:
            context.emit(
                "consistency_postprocess",
                "done",
                "Removed model-generated warnings contradicted by validator math.",
                actions=warning_actions,
            )
            context.postprocess_actions.extend(warning_actions)
            save_json(
                paths["consistency_postprocess"],
                {"actions": context.postprocess_actions},
            )

        save_json(paths["validation_report"], final_report)
        save_json(paths["reconciliation_report"], final_report)

        llm_result = context.llm_result
        context.output_receipt = dict(final_receipt)
        context.output_receipt["validation"] = final_report
        context.output_receipt["pipeline"] = {
            "architecture": get_app_version(),
            "app_version": get_app_version(),
            "workflow": "ReceiptExtractionWorkflow",
            "staged_execution": True,
            "extraction_strategy": config.extraction_strategy,
            "spatial_overview_used": bool(llm_result.get("spatial_overview_used")),
            "spatial_geometry_used": bool(llm_result.get("spatial_geometry_used")),
            "spatial_overview_llm_call_performed": bool(
                (context.spatial_overview_result or {}).get("llm_call_performed")
            ),
            "response_schema_enforced": bool(llm_result.get("response_schema_enforced")),
            "no_deterministic_fallback": True,
            "llm_error": llm_result.get("error"),
            "vlm_enabled": config.vlm_enabled,
            "vlm_first": bool(config.vlm_enabled),
            "region_reocr_first": bool(config.vlm_enabled and config.source_image_path),
            "visual_evidence_used_by_main_llm": bool(llm_result.get("visual_evidence_used")),
            "vlm_status": (
                context.visual_result.get("status")
                if context.visual_result
                else ("disabled" if not config.vlm_enabled else "skipped")
            ),
            "correction_used": context.correction_used,
            "consistency_postprocess_actions": context.postprocess_actions,
        }
        save_json(paths["receipt_final"], context.output_receipt)
        save_json(paths["receipt_final_reconciled"], context.output_receipt)

        self._categorize(context)
        context.pipeline_meta = self._build_pipeline_meta(context)
        save_json(paths["pipeline_meta"], context.pipeline_meta)
        save_json(paths["stage_trace"], context.stage_trace)
        publish_latest_aliases(paths, config.result_dir)

        context.emit(
            "pipeline",
            "done",
            f"{get_app_version()} staged pipeline finished.",
            initial_decision=context.initial_report.get("import_decision"),
            final_decision=final_report.get("import_decision"),
            correction_used=context.correction_used,
            balanced=final_report.get("balanced"),
            difference=final_report.get("difference"),
            issue_count=len(final_report.get("issues") or []),
            duration_seconds=context.duration_seconds,
            stages=[entry["stage"] for entry in context.stage_trace],
        )
        return context

    def _categorize(self, context: ExtractionContext) -> None:
        config = context.config
        output_receipt = context.output_receipt
        context.categorized_receipt = output_receipt
        if config.categorization_enabled:
            context.emit(
                "item_categorization",
                "running",
                "Running V14.14 LLM-first item categorization on final receipt items.",
            )
            try:
                context.categorization_result = categorize_receipt_items_llm(
                    output_receipt,
                    ollama_url=config.ollama_url,
                    model=config.categorization_model or config.model,
                    num_ctx=config.categorization_num_ctx,
                    num_predict=config.categorization_num_predict,
                    keep_alive=config.keep_alive,
                    timeout=config.categorization_timeout_seconds,
                    format_json=config.categorization_format_json,
                    llm_gateway=context.dependencies.llm_gateway,
                )
                category_paths = write_categorization_artifacts(
                    context.categorization_result,
                    result_dir=config.result_dir,
                    run_id=config.run_id,
                )
                context.paths.update(category_paths)
                context.categorized_receipt = (
                    context.categorization_result.get("receipt") or output_receipt
                )
            except Exception as exc:
                context.categorization_result = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "receipt": output_receipt,
                    "duration_seconds": None,
                }
                context.categorized_receipt = output_receipt
                try:
                    category_paths = write_categorization_artifacts(
                        context.categorization_result,
                        result_dir=config.result_dir,
                        run_id=config.run_id,
                    )
                    context.paths.update(category_paths)
                except Exception:
                    save_json(
                        context.paths["receipt_final_categorized"],
                        context.categorized_receipt,
                    )
            status = context.categorization_result.get("status")
            categorization = context.categorized_receipt.get("categorization")
            context.emit(
                "item_categorization",
                "done" if status in {"ok", "ok_with_warnings", "skipped_no_items"} else "error",
                "Item categorization finished.",
                categorization_status=status,
                item_count=(
                    categorization.get("item_count") if isinstance(categorization, dict) else None
                ),
                categorized_count=(
                    categorization.get("categorized_count")
                    if isinstance(categorization, dict)
                    else None
                ),
                duration_seconds=context.categorization_result.get("duration_seconds"),
                error=context.categorization_result.get("error"),
            )
            save_json(context.paths["receipt_final"], context.categorized_receipt)
            save_json(context.paths["receipt_final_reconciled"], context.categorized_receipt)
        else:
            context.emit("item_categorization", "skipped", "Item categorization disabled.")
            save_json(context.paths["receipt_final_categorized"], context.categorized_receipt)

    def _build_pipeline_meta(self, context: ExtractionContext) -> dict[str, Any]:
        config = context.config
        llm_result = context.llm_result
        initial_report = context.initial_report
        final_report = context.final_report
        categorization = (
            context.categorized_receipt.get("categorization")
            if context.categorized_receipt
            else None
        )
        return {
            "schema_version": "v14_6_pipeline_meta_1",
            "app_version": get_app_version(),
            "workflow": {
                "name": "ReceiptExtractionWorkflow",
                "stage_count": len(context.stage_trace),
                "stages": [entry["stage"] for entry in context.stage_trace],
                "trace_artifact": str(context.paths["stage_trace"]),
                "metrics_artifact": str(context.paths["extraction_metrics"]),
            },
            "architecture": (
                "OCR/VLM evidence -> strategy-selected geometry-preserving representation -> "
                "schema-constrained main LLM parser -> validation -> strategy-gated repair -> "
                "validated patch-only correction -> item categorization"
            ),
            "extraction_strategy": config.extraction_strategy,
            "spatial_overview": {
                "enabled": config.extraction_strategy == "spatial_overview",
                "status": (
                    context.spatial_overview_result.get("status")
                    if context.spatial_overview_result
                    else None
                ),
                "mode": (context.spatial_overview_result or {}).get("mode"),
                "llm_call_performed": bool(
                    (context.spatial_overview_result or {}).get("llm_call_performed")
                ),
                "geometric_row_group_count": (
                    context.spatial_overview_result or {}
                ).get("geometric_row_group_count"),
                "document_map_artifact": str(context.paths["spatial_document_map"]),
                "overview_artifact": str(context.paths["spatial_overview"]),
            },
            "no_deterministic_semantic_parser": True,
            "no_deterministic_fallback": True,
            "old_v13_arguments_ignored": {
                "skip_row_llm": config.skip_row_llm,
                "active_line_repair": config.active_line_repair,
                "max_repair_passes": config.max_repair_passes,
                "max_repair_rois": config.max_repair_rois,
                "max_repair_variants": config.max_repair_variants,
                "max_reocr_images": config.max_reocr_images,
                "repair_time_budget_seconds": config.repair_time_budget_seconds,
                "repair_ocr_min_score": config.repair_ocr_min_score,
                "ocr_lang": config.ocr_lang,
                "ocr_device": config.ocr_device,
                "ocr_det_model": config.ocr_det_model,
                "ocr_rec_model": config.ocr_rec_model,
            },
            "llm": {
                "model": config.model,
                "ollama_url": config.ollama_url,
                "num_ctx": config.num_ctx,
                "num_predict": config.num_predict,
                "keep_alive": config.keep_alive,
                "timeout_seconds": config.llm_timeout_seconds,
                "error": llm_result.get("error"),
                "duration_seconds": llm_result.get("duration_seconds"),
                "json_retry_count": config.json_retry_count,
                "format_json": config.format_json,
                "response_schema_enforced": bool(
                    llm_result.get("response_schema_enforced")
                ),
                "attempts": llm_result.get("attempts"),
            },
            "right_column_reocr": self._reocr_meta(context),
            "right_column_recovery": self._right_column_meta(context),
            "vertical_price_stack_recovery": self._vertical_stack_meta(context),
            "vlm": {
                "enabled": config.vlm_enabled,
                "backend": config.vlm_backend,
                "vlm_first": bool(config.vlm_enabled),
                "region_reocr_first": bool(config.vlm_enabled and config.source_image_path),
                "triggered": context.visual_result is not None,
                "used_by_main_llm": bool(llm_result.get("visual_evidence_used")),
                "status": context.visual_result.get("status") if context.visual_result else None,
                "error": context.visual_result.get("error") if context.visual_result else None,
                "source_image_path": str(config.source_image_path)
                if config.source_image_path
                else None,
            },
            "table_interpretation": self._table_interpretation_meta(context),
            "table_arbitration": self._table_arbitration_meta(context),
            "table_assembly": context.table_assembly_report,
            "consistency_postprocess": {
                "actions": context.postprocess_actions,
                "action_count": len(context.postprocess_actions),
            },
            "correction": self._correction_meta(context),
            "validation": {
                "initial_import_decision": initial_report.get("import_decision"),
                "final_import_decision": final_report.get("import_decision"),
                "balanced": final_report.get("balanced"),
                "difference": final_report.get("difference"),
                "issue_count": len(final_report.get("issues") or []),
                "failure_diagnosis": final_report.get("failure_diagnosis"),
            },
            "categorization": {
                "enabled": config.categorization_enabled,
                "status": (
                    context.categorization_result.get("status")
                    if context.categorization_result
                    else ("disabled" if not config.categorization_enabled else None)
                ),
                "model": config.categorization_model or config.model,
                "duration_seconds": (
                    context.categorization_result.get("duration_seconds")
                    if context.categorization_result
                    else None
                ),
                "error": (
                    context.categorization_result.get("error")
                    if context.categorization_result
                    else None
                ),
                "item_count": categorization.get("item_count")
                if isinstance(categorization, dict)
                else None,
                "categorized_count": (
                    categorization.get("categorized_count")
                    if isinstance(categorization, dict)
                    else None
                ),
            },
            "duration_seconds": context.duration_seconds,
        }

    @staticmethod
    def _reocr_meta(context: ExtractionContext) -> dict[str, Any]:
        result = context.reocr_result
        return {
            "attempted": result is not None,
            "status": result.get("status") if result else None,
            "evidence_line_count": result.get("evidence_line_count") if result else None,
        }

    @staticmethod
    def _right_column_meta(context: ExtractionContext) -> dict[str, Any]:
        result = context.right_column_recovery_result
        return {
            "attempted": result is not None,
            "status": result.get("status") if result else None,
            "applied": bool(result.get("applied")) if result else False,
            "before_diff": result.get("before_diff") if result else None,
            "after_diff": result.get("after_diff") if result else None,
            "selected_addition_count": len(result.get("selected_additions") or []) if result else 0,
            "replacement_count": len(result.get("replacement_actions") or []) if result else 0,
        }

    @staticmethod
    def _vertical_stack_meta(context: ExtractionContext) -> dict[str, Any]:
        result = context.vertical_price_stack_recovery_result
        return {
            "attempted": result is not None,
            "status": result.get("status") if result else None,
            "applied": bool(result.get("applied")) if result else False,
            "mode": result.get("mode") if result else None,
            "before_diff": result.get("before_diff") if result else None,
            "after_diff": result.get("after_diff") if result else None,
            "candidate_item_count": result.get("candidate_item_count") if result else 0,
            "amount_count": (
                (result.get("price_stack_crop") or {}).get("amount_count") if result else None
            ),
        }

    @staticmethod
    def _table_interpretation_meta(context: ExtractionContext) -> dict[str, Any]:
        result = context.table_interpretation_result
        return {
            "attempted": result is not None,
            "status": result.get("status") if result else None,
            "table_count": len(result.get("tables") or []) if result else 0,
            "overall_confidence": result.get("overall_confidence") if result else None,
            "duration_seconds": result.get("duration_seconds") if result else None,
            "warnings": result.get("warnings") if result else [],
        }

    @staticmethod
    def _table_arbitration_meta(context: ExtractionContext) -> dict[str, Any]:
        result = context.table_arbitration_result
        return {
            "attempted": result is not None,
            "status": result.get("status") if result else None,
            "summary": result.get("summary") if result else None,
            "warning_count": len(result.get("warnings") or []) if result else 0,
        }

    @staticmethod
    def _correction_meta(context: ExtractionContext) -> dict[str, Any]:
        result = context.patch_correction_result
        return {
            "enabled": context.config.correction_enabled,
            "mode": "patch_only",
            "full_receipt_rewrite_enabled": False,
            "patch_attempted": result is not None,
            "patch_status": result.get("status") if result else None,
            "patch_count": len(result.get("patches") or []) if result else 0,
            "patch_attempt_count": result.get("attempt_count") if result else 0,
            "patch_retry_used": bool(result.get("retry_used")) if result else False,
            "used": context.correction_used,
            "warnings": result.get("warnings") if result else [],
            "corrected_report": context.corrected_report,
        }
