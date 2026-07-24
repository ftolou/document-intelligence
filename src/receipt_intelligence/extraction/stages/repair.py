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
from receipt_intelligence.extraction.repair.patch_correction import (
    apply_correction_patches,
    run_patch_correction_pass,
)
from receipt_intelligence.extraction.repair.reocr import (
    reocr_evidence_to_visual_evidence,
    run_bounded_right_column_reocr,
)
from receipt_intelligence.extraction.repair.right_column import (
    run_right_column_recovery,
)
from receipt_intelligence.extraction.repair.vertical_price_stack import (
    run_vertical_price_stack_recovery,
)
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

    def run(self, context: ExtractionContext) -> ExtractionContext:
        config = context.config
        report = context.require("report")
        llm_result = context.require("llm_result")

        if (
            config.correction_enabled
            and should_run_visual_layer(report)
            and not llm_result.get("error")
        ):
            self._run_bounded_reocr(context)
            self._run_right_column_recovery(context)
            self._run_vertical_price_stack_recovery(context)
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

    def _run_bounded_reocr(self, context: ExtractionContext) -> None:
        config = context.config
        context.emit(
            "right_column_reocr",
            "running",
            (
                "Validation did not pass cleanly; running bounded right-column re-OCR "
                "evidence pass before optional VLM."
            ),
            decision=context.require("report").get("import_decision"),
        )
        context.reocr_result = run_bounded_right_column_reocr(
            image_path=config.source_image_path,
            ocr_context=context.require("ocr_context"),
            validation_report=context.require("report"),
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
                context.require("report"),
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

    def _run_right_column_recovery(self, context: ExtractionContext) -> None:
        config = context.config
        context.emit(
            "right_column_recovery",
            "running",
            (
                "Checking whether bounded right-column re-OCR can safely improve "
                "item-total reconciliation."
            ),
        )
        context.right_column_recovery_result = run_right_column_recovery(
            receipt=context.require("receipt"),
            validation_report=context.require("report"),
            ocr_context=context.require("ocr_context"),
            reocr_result=context.reocr_result,
            table_arbitration=context.table_arbitration_result,
            tolerance=config.tolerance,
        )
        result = context.right_column_recovery_result
        save_json(
            context.paths["right_column_recovery"],
            {key: value for key, value in result.items() if key != "receipt"},
        )
        if result.get("applied") and isinstance(result.get("receipt"), dict):
            recovered_receipt = result["receipt"]
            save_json(context.paths["receipt_right_column_recovered"], recovered_receipt)
            recovered_report = validate_receipt(
                recovered_receipt,
                context.require("ocr_context"),
                tolerance=config.tolerance,
            )
            save_json(
                context.paths["validation_report_right_column_recovered"],
                recovered_report,
            )
            if report_score(recovered_report) > report_score(context.require("report")):
                self._select_candidate(context, recovered_receipt, recovered_report)
                context.emit(
                    "right_column_recovery",
                    "done",
                    "Right-column recovery improved validation and was selected.",
                    before_difference=result.get("before_diff"),
                    after_difference=result.get("after_diff"),
                    selected_addition_count=len(result.get("selected_additions") or []),
                    replacement_count=len(result.get("replacement_actions") or []),
                )
            else:
                context.emit(
                    "right_column_recovery",
                    "done",
                    (
                        "Right-column recovery produced a candidate receipt but did not "
                        "improve validation enough; original receipt kept."
                    ),
                    status_detail=result.get("status"),
                )
        else:
            context.emit(
                "right_column_recovery",
                result.get("status", "done"),
                "Right-column recovery did not apply changes.",
                reason=result.get("reason"),
                before_difference=result.get("before_diff"),
            )

    def _run_vertical_price_stack_recovery(self, context: ExtractionContext) -> None:
        if not should_run_visual_layer(context.require("report")):
            return
        config = context.config
        context.emit(
            "vertical_price_stack_recovery",
            "running",
            (
                "Checking whether a full right-side price-stack crop can safely improve "
                "item-total reconciliation."
            ),
        )
        context.vertical_price_stack_recovery_result = run_vertical_price_stack_recovery(
            receipt=context.require("receipt"),
            validation_report=context.require("report"),
            ocr_context=context.require("ocr_context"),
            image_path=config.source_image_path,
            result_dir=config.result_dir,
            run_id=config.run_id,
            lang=config.ocr_lang,
            device=config.ocr_device,
            min_score=config.repair_ocr_min_score or 0.20,
            tolerance=config.tolerance,
            visual_evidence=context.visual_evidence,
            table_arbitration=context.table_arbitration_result,
            progress_callback=config.progress_callback,
        )
        result = context.vertical_price_stack_recovery_result
        save_json(
            context.paths["vertical_price_stack_recovery"],
            {key: value for key, value in result.items() if key != "receipt"},
        )
        if result.get("applied") and isinstance(result.get("receipt"), dict):
            recovered_receipt = result["receipt"]
            save_json(context.paths["receipt_vertical_price_stack_recovered"], recovered_receipt)
            recovered_report = validate_receipt(
                recovered_receipt,
                context.require("ocr_context"),
                tolerance=config.tolerance,
            )
            save_json(
                context.paths["validation_report_vertical_price_stack_recovered"],
                recovered_report,
            )
            if report_score(recovered_report) > report_score(context.require("report")):
                self._select_candidate(context, recovered_receipt, recovered_report)
                context.emit(
                    "vertical_price_stack_recovery",
                    "done",
                    "Vertical price-stack recovery improved validation and was selected.",
                    before_difference=result.get("before_diff"),
                    after_difference=result.get("after_diff"),
                    mode=result.get("mode"),
                )
            else:
                context.emit(
                    "vertical_price_stack_recovery",
                    "done",
                    (
                        "Vertical price-stack recovery produced a candidate receipt but did "
                        "not improve validation enough; original receipt kept."
                    ),
                    status_detail=result.get("status"),
                )
        else:
            context.emit(
                "vertical_price_stack_recovery",
                result.get("status", "done"),
                "Vertical price-stack recovery did not apply changes.",
                reason=result.get("reason"),
                before_difference=result.get("before_diff"),
            )

    def _run_late_vlm_if_needed(self, context: ExtractionContext) -> None:
        config = context.config
        if config.vlm_enabled and context.visual_result is None:
            self._unload_ollama_if_requested(context)
            context.emit(
                "visual_evidence",
                "running",
                "Validation did not pass cleanly; running optional VLM visual evidence layer.",
                decision=context.require("report").get("import_decision"),
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
                    context.require("report"),
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
            previous_receipt=context.require("receipt"),
            validation_report=context.require("report"),
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

        selected = False
        patch_actions: list[dict[str, Any]] = []
        patch_postprocess_actions: list[dict[str, Any]] = []
        if result.get("status") == "ok" and result.get("patches"):
            patch_receipt, patch_actions = apply_correction_patches(
                context.require("receipt"),
                result,
            )
            patch_receipt, patch_postprocess_actions = apply_consistency_postprocess(
                patch_receipt,
                context.visual_evidence,
                context.require("ocr_context"),
                tolerance=max(config.tolerance, 0.05),
            )
            save_json(context.paths["receipt_patch_corrected"], patch_receipt)
            patch_report = validate_receipt(
                patch_receipt,
                context.require("ocr_context"),
                tolerance=config.tolerance,
            )
            save_json(context.paths["validation_report_patch_corrected"], patch_report)
            context.corrected_report = patch_report
            if report_score(patch_report) > report_score(context.require("report")):
                self._select_candidate(context, patch_receipt, patch_report)
                selected = True
                context.emit(
                    "llm_patch_correction",
                    "done",
                    "Patch correction improved validation and was selected.",
                    before=context.require("report").get("import_decision"),
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
                    before=context.require("report").get("import_decision"),
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

        if not selected:
            save_json(
                context.paths["receipt_llm_corrected"],
                {
                    "status": "skipped",
                    "mode": "patch_only",
                    "reason": "full_receipt_rewrite_disabled",
                    "receipt_unchanged": True,
                    "patch_status": result.get("status"),
                    "patch_count": len(result.get("patches") or []),
                    "patch_warnings": result.get("warnings") or [],
                },
            )
            write_text(
                context.paths["correction_prompt"],
                "Full receipt correction rewrite disabled. Patch-only correction is active.\n",
            )
            write_text(context.paths["correction_raw"], "")
            context.emit(
                "llm_correction",
                "skipped",
                (
                    "Full receipt JSON correction rewrite disabled; patch-only result did not "
                    "improve validation, so the original receipt was kept."
                ),
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
