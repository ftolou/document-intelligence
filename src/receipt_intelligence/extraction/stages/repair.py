"""Validation-gated OCR recovery and patch-only LLM correction."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.application.ports import ModelLifecycleRequest, VlmRequest
from receipt_intelligence.extraction.artifacts import save_json, write_text
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.evidence.visual import (
    build_visual_evidence,
    visual_evidence_to_prompt_text,
)
from receipt_intelligence.extraction.repair.line_price_fusion import (
    repair_receipt_line_prices,
)
from receipt_intelligence.extraction.repair.patch_correction import (
    apply_correction_patches,
    run_patch_correction_pass,
)
from receipt_intelligence.extraction.repair.reocr import (
    reocr_evidence_to_visual_evidence,
    run_bounded_right_column_reocr,
)
from receipt_intelligence.extraction.state import ExtractionPhase
from receipt_intelligence.extraction.support import (
    merge_visual_evidence,
    report_score,
    should_run_visual_layer,
)
from receipt_intelligence.extraction.validation.consistency import (
    apply_consistency_postprocess,
)
from receipt_intelligence.extraction.validation.receipt import validate_receipt


class RepairAndCorrectionStage:
    name = "repair_and_correction"
    input_phase = ExtractionPhase.PARSED
    output_phase = ExtractionPhase.REPAIRED

    def run(self, context: ExtractionContext) -> ExtractionContext:
        context.begin_repair_stage()
        config = context.config
        llm_result = context.llm_result

        self._run_spatial_line_price_fusion(context)
        report = context.report

        if (
            config.correction_enabled
            and should_run_visual_layer(report)
            and not llm_result.get("error")
        ):
            self._run_bounded_reocr(context)
            self._run_late_vlm_if_needed(context)
            self._run_patch_correction(context)
        elif not config.correction_enabled:
            save_json(
                context.paths["vlm_raw_output"],
                {"status": "skipped", "message": "Correction/VLM layer disabled."},
            )
        elif not config.vlm_enabled:
            save_json(
                context.paths["vlm_raw_output"],
                {"status": "disabled", "message": "VLM layer disabled."},
            )
        else:
            save_json(
                context.paths["vlm_raw_output"],
                {"status": "skipped", "message": "VLM layer not triggered."},
            )
        return context

    def _run_spatial_line_price_fusion(self, context: ExtractionContext) -> None:
        candidate_receipt, actions = repair_receipt_line_prices(
            context.receipt,
            context.report,
            context.spatial_document_map,
            tolerance=context.config.tolerance,
        )
        if not actions:
            save_json(
                context.paths["line_price_fusion"],
                {
                    "status": "no_change",
                    "action_count": 0,
                    "message": (
                        "No high-confidence region item-price candidate required a field-level "
                        "repair."
                    ),
                },
            )
            return

        candidate_report = validate_receipt(
            candidate_receipt,
            context.ocr_context,
            tolerance=context.config.tolerance,
        )
        save_json(context.paths["receipt_line_price_fused"], candidate_receipt)
        save_json(context.paths["validation_report_line_price_fused"], candidate_report)
        selected = report_score(candidate_report) > report_score(context.report)
        save_json(
            context.paths["line_price_fusion"],
            {
                "status": "selected" if selected else "rejected",
                "action_count": len(actions),
                "actions": actions,
                "before": {
                    "import_decision": context.report.get("import_decision"),
                    "difference": context.report.get("difference"),
                },
                "after": {
                    "import_decision": candidate_report.get("import_decision"),
                    "difference": candidate_report.get("difference"),
                },
            },
        )
        if selected:
            self._select_candidate(context, candidate_receipt, candidate_report)
            context.emit(
                "line_price_fusion",
                "done",
                (
                    "Selected field-level item price repairs from high-confidence region crop "
                    "OCR evidence."
                ),
                action_count=len(actions),
                before=context.initial_report.get("import_decision"),
                after=candidate_report.get("import_decision"),
                difference=candidate_report.get("difference"),
            )
        else:
            context.emit(
                "line_price_fusion",
                "done",
                "Region item-price patches did not improve validation and were not selected.",
                action_count=len(actions),
            )

    def _run_bounded_reocr(self, context: ExtractionContext) -> None:
        config = context.config
        context.emit(
            "right_column_reocr",
            "running",
            (
                "Validation did not pass cleanly; running bounded right-column re-OCR "
                "evidence pass before optional VLM."
            ),
            decision=context.report.get("import_decision"),
        )
        context.reocr_result = run_bounded_right_column_reocr(
            image_path=config.source_image_path,
            ocr_context=context.ocr_context,
            validation_report=context.report,
            result_dir=config.result_dir,
            run_id=config.run_id,
            enabled=True,
            max_crops=config.max_reocr_images or 8,
            lang=config.ocr_lang,
            device=config.ocr_device,
            min_score=config.repair_ocr_min_score or 0.20,
            progress_callback=config.progress_callback,
        )
        save_json(context.paths["right_column_reocr"], context.reocr_result)
        if context.reocr_result.get("status") == "ok":
            reocr_evidence = reocr_evidence_to_visual_evidence(
                context.reocr_result,
                context.report,
            )
            context.visual_evidence = merge_visual_evidence(
                context.visual_evidence,
                reocr_evidence,
                backend_suffix="bounded_right_column_reocr",
            )
            save_json(context.paths["visual_evidence"], context.visual_evidence)
            write_text(
                context.paths["visual_evidence_text"],
                visual_evidence_to_prompt_text(context.visual_evidence),
            )

    def _run_late_vlm_if_needed(self, context: ExtractionContext) -> None:
        config = context.config
        if config.vlm_enabled and context.visual_result is None:
            self._unload_ollama_if_requested(context)
            context.emit(
                "visual_evidence",
                "running",
                "Validation did not pass cleanly; running optional VLM visual evidence layer.",
                decision=context.report.get("import_decision"),
            )
            context.visual_result = context.dependencies.vlm_engine.analyze(
                VlmRequest(
                    image_path=config.source_image_path,
                    result_dir=config.result_dir,
                    run_id=config.run_id,
                    enabled=config.vlm_enabled,
                    timeout_seconds=config.vlm_timeout_seconds,
                    progress_callback=config.progress_callback,
                )
            )
            self._reload_ollama_if_requested(context)
            save_json(context.paths["vlm_raw_output"], context.visual_result)
            if context.visual_result.get("status") == "ok":
                vlm_evidence = build_visual_evidence(
                    context.visual_result,
                    context.report,
                    max_chars=config.vlm_max_chars,
                )
                context.visual_evidence = merge_visual_evidence(
                    context.visual_evidence,
                    vlm_evidence,
                    backend_suffix="paddleocr_vl_structured_tables",
                )
                save_json(context.paths["visual_evidence"], context.visual_evidence)
                write_text(
                    context.paths["visual_evidence_text"],
                    visual_evidence_to_prompt_text(context.visual_evidence),
                )
            else:
                context.emit(
                    "visual_evidence",
                    "done",
                    (
                        "VLM visual layer did not produce usable evidence; bounded re-OCR/"
                        "original LLM output kept."
                    ),
                    vlm_status=context.visual_result.get("status"),
                    error=context.visual_result.get("error"),
                )
        elif not config.vlm_enabled:
            save_json(
                context.paths["vlm_raw_output"],
                {
                    "status": "disabled",
                    "message": "VLM layer disabled; bounded right-column re-OCR may still run.",
                },
            )

        if config.vlm_enabled and context.visual_result is not None:
            context.emit(
                "visual_evidence",
                "done",
                (
                    "Reusing VLM-first visual evidence for the correction pass; VLM was not "
                    "run a second time."
                ),
                vlm_status=context.visual_result.get("status"),
            )

    def _run_patch_correction(self, context: ExtractionContext) -> None:
        evidence = context.visual_evidence or {}
        usable = evidence.get("status") in {"ok", "no_amounts_found"} and any(
            evidence.get(key)
            for key in (
                "amount_lines",
                "item_price_like_lines",
                "payment_change_lines",
                "tax_like_lines",
                "structured_tables",
                "quantity_hint_rows",
            )
        )
        if not usable:
            context.emit(
                "llm_correction",
                "done",
                "No usable extra visual/OCR evidence; original LLM output kept.",
            )
            return

        config = context.config
        context.emit(
            "llm_patch_correction",
            "running",
            "Running compact patch-only LLM correction; full receipt rewrite is disabled.",
        )
        context.patch_correction_result = run_patch_correction_pass(
            previous_receipt=context.receipt,
            validation_report=context.report,
            visual_evidence=evidence,
            ollama_url=config.ollama_url,
            model=config.model,
            num_ctx=min(config.num_ctx, 18432),
            num_predict=min(max(config.num_predict, 1024), 2048),
            keep_alive=config.keep_alive,
            timeout=min(max(config.llm_timeout_seconds, 180.0), 240.0),
            format_json=config.format_json,
            llm_gateway=context.dependencies.llm_gateway,
        )
        result = context.patch_correction_result
        write_text(context.paths["correction_patch_prompt"], result.get("prompt") or "")
        write_text(context.paths["correction_patch_raw"], result.get("raw_output") or "")
        save_json(
            context.paths["correction_patch_result"],
            {key: value for key, value in result.items() if key not in {"prompt", "raw_output"}},
        )

        patch_actions: list[dict[str, Any]] = []
        patch_postprocess_actions: list[dict[str, Any]] = []
        if result.get("status") == "ok" and result.get("patches"):
            patch_receipt, patch_actions = apply_correction_patches(
                context.receipt,
                result,
            )
            patch_receipt, patch_postprocess_actions = apply_consistency_postprocess(
                patch_receipt,
                context.visual_evidence,
                context.ocr_context,
                tolerance=max(config.tolerance, 0.05),
            )
            save_json(context.paths["receipt_patch_corrected"], patch_receipt)
            patch_report = validate_receipt(
                patch_receipt,
                context.ocr_context,
                tolerance=config.tolerance,
            )
            save_json(context.paths["validation_report_patch_corrected"], patch_report)
            context.corrected_report = patch_report
            if report_score(patch_report) > report_score(context.report):
                self._select_candidate(context, patch_receipt, patch_report)
                context.emit(
                    "llm_patch_correction",
                    "done",
                    "Patch correction improved validation and was selected.",
                    before=context.report.get("import_decision"),
                    after=patch_report.get("import_decision"),
                    applied_patch_count=len(patch_actions),
                    postprocess_action_count=len(patch_postprocess_actions),
                )
            else:
                context.emit(
                    "llm_patch_correction",
                    "done",
                    (
                        "Patch correction did not improve validation; original receipt kept. "
                        "Full receipt rewrite is disabled."
                    ),
                    before=context.report.get("import_decision"),
                    after=patch_report.get("import_decision"),
                    applied_patch_count=len(patch_actions),
                )
        else:
            context.emit(
                "llm_patch_correction",
                result.get("status", "done"),
                (
                    "Patch correction returned no usable patches; original receipt kept. "
                    "Full receipt rewrite is disabled."
                ),
                warning_count=len(result.get("warnings") or []),
            )


    @staticmethod
    def _select_candidate(
        context: ExtractionContext,
        receipt: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        context.receipt = receipt
        context.report = report
        context.final_receipt = receipt
        context.final_report = report
        context.correction_used = True

    def _unload_ollama_if_requested(self, context: ExtractionContext) -> None:
        config = context.config
        if (
            config.gpu_orchestration
            not in {"sequential", "sequential_ollama_handoff", "ollama_handoff"}
            or not config.unload_llm_before_vlm
        ):
            return
        context.emit(
            "gpu_orchestration",
            "running",
            "Sequential GPU mode: unloading Ollama/Gemma before VLM to free GPU memory.",
            mode=config.gpu_orchestration,
            control_mode=config.ollama_control_mode,
        )
        result = context.dependencies.model_lifecycle.release_for_vlm(
            ModelLifecycleRequest(
                model=config.model,
                timeout_seconds=config.ollama_control_timeout_seconds,
                wait_seconds=config.ollama_gpu_handoff_wait_seconds,
            )
        )
        save_json(context.paths["gpu_orchestration_before_vlm"], result)

    def _reload_ollama_if_requested(self, context: ExtractionContext) -> None:
        config = context.config
        if (
            config.gpu_orchestration
            not in {"sequential", "sequential_ollama_handoff", "ollama_handoff"}
            or not config.reload_llm_after_vlm
        ):
            return
        context.emit(
            "gpu_orchestration",
            "running",
            "Sequential GPU mode: reloading Ollama/Gemma after VLM for correction.",
            mode=config.gpu_orchestration,
            control_mode=config.ollama_control_mode,
        )
        result = context.dependencies.model_lifecycle.restore_after_vlm(
            ModelLifecycleRequest(
                model=config.model,
                keep_alive=config.keep_alive,
                timeout_seconds=config.ollama_control_timeout_seconds,
                warmup_prompt=config.ollama_reload_prompt,
                wait_seconds=config.ollama_gpu_handoff_wait_seconds,
            )
        )
        save_json(context.paths["gpu_orchestration_after_vlm"], result)
