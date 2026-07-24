"""Build optional VLM, region re-OCR and table evidence before parsing."""

from __future__ import annotations

from receipt_intelligence.application.ports import ModelLifecycleRequest, VlmRequest
from receipt_intelligence.extraction.artifacts import save_json, write_text
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.evidence.visual import (
    build_visual_evidence,
    visual_evidence_to_prompt_text,
)
from receipt_intelligence.extraction.parsing.table_arbitration import (
    attach_table_arbitration_to_visual_evidence,
    build_table_arbitration,
)
from receipt_intelligence.extraction.parsing.table_interpreter import (
    attach_table_interpretation_to_visual_evidence,
    run_table_interpreter,
)
from receipt_intelligence.extraction.repair.region_reocr import (
    merge_region_reocr_into_visual_evidence,
    run_vlm_region_reocr,
)
from receipt_intelligence.extraction.state import ExtractionPhase


class VisualEvidenceStage:
    name = "visual_evidence"
    input_phase = ExtractionPhase.PREPARED
    output_phase = ExtractionPhase.VISUAL_READY

    def run(self, context: ExtractionContext) -> ExtractionContext:
        context.begin_visual_stage()
        config = context.config
        paths = context.paths
        emit = context.emit

        if config.vlm_enabled and config.source_image_path:
            self._unload_ollama_if_requested(context)
            emit(
                "visual_evidence",
                "running",
                (
                    "VLM-first mode: running PaddleOCR-VL before the main LLM parser "
                    "to build structured table evidence."
                ),
                backend=config.vlm_backend,
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
            save_json(paths["vlm_raw_output"], context.visual_result)
            if context.visual_result.get("status") == "ok":
                context.visual_evidence = build_visual_evidence(
                    context.visual_result,
                    {"import_decision": "pre_llm_vlm_first", "issues": []},
                    max_chars=config.vlm_max_chars,
                )
                self._run_region_reocr(context)
                self._run_table_interpretation(context)
                self._run_table_arbitration(context)
                save_json(paths["visual_evidence"], context.visual_evidence)
                write_text(
                    paths["visual_evidence_text"],
                    visual_evidence_to_prompt_text(context.visual_evidence),
                )
                emit(
                    "visual_evidence",
                    "done",
                    "VLM-region-first evidence is ready for the main LLM parser.",
                    summary=context.visual_evidence.get("summary"),
                )
            else:
                emit(
                    "visual_evidence",
                    "done",
                    "VLM-first layer did not produce usable evidence; using OCR-only LLM prompt.",
                    vlm_status=context.visual_result.get("status"),
                    error=context.visual_result.get("error"),
                )
            self._reload_ollama_if_requested(context)
        elif config.vlm_enabled:
            save_json(
                paths["vlm_raw_output"],
                {"status": "skipped", "message": "VLM enabled but source_image_path was missing."},
            )
        else:
            save_json(
                paths["vlm_raw_output"],
                {"status": "disabled", "message": "VLM layer disabled."},
            )
        return context

    def _run_region_reocr(self, context: ExtractionContext) -> None:
        config = context.config
        paths = context.paths
        try:
            context.emit(
                "region_reocr",
                "running",
                "Running OCR on VLM-detected layout regions from the original image.",
            )
            context.region_reocr_result = run_vlm_region_reocr(
                source_image_path=config.source_image_path,
                vlm_result=context.visual_result,
                visual_evidence=context.visual_evidence,
                result_dir=config.result_dir,
                run_id=config.run_id,
                lang=config.ocr_lang,
                device=config.ocr_device,
                max_regions=3,
            )
            save_json(paths["region_reocr"], context.region_reocr_result)
            context.visual_evidence = merge_region_reocr_into_visual_evidence(
                context.visual_evidence,
                context.region_reocr_result,
            )
            context.emit(
                "region_reocr",
                context.region_reocr_result.get("status", "done"),
                "VLM-region crop re-OCR finished.",
                summary={
                    "selected_region_count": context.region_reocr_result.get(
                        "selected_region_count"
                    ),
                    "preferred_item_block_count": len(
                        context.region_reocr_result.get("preferred_item_blocks") or []
                    ),
                    "best_block_balanced": bool(
                        (context.region_reocr_result.get("best_preferred_item_block") or {}).get(
                            "balanced_to_printed_total"
                        )
                    ),
                },
            )
        except Exception as exc:
            context.region_reocr_result = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            save_json(paths["region_reocr"], context.region_reocr_result)
            context.emit(
                "region_reocr",
                "error",
                "VLM-region crop re-OCR failed; continuing with VLM/OCR evidence.",
                error=context.region_reocr_result["error"],
            )

    def _run_table_interpretation(self, context: ExtractionContext) -> None:
        visual_evidence = context.visual_evidence or {}
        if not visual_evidence.get("structured_tables"):
            return
        config = context.config
        context.emit(
            "table_interpretation",
            "running",
            "Running dedicated LLM table interpretation before the main parser.",
            structured_table_count=len(visual_evidence.get("structured_tables") or []),
        )
        context.table_interpretation_result = run_table_interpreter(
            visual_evidence=visual_evidence,
            ollama_url=config.ollama_url,
            model=config.model,
            num_ctx=min(config.num_ctx, 18432),
            num_predict=max(min(config.num_predict, 8192), 8192),
            keep_alive=config.keep_alive,
            timeout=min(max(config.llm_timeout_seconds, 180.0), 300.0),
            format_json=config.format_json,
            llm_gateway=context.dependencies.llm_gateway,
        )
        result = context.table_interpretation_result
        save_json(
            context.paths["table_interpretation"],
            {key: value for key, value in result.items() if key not in {"prompt", "raw_output"}},
        )
        write_text(context.paths["table_interpretation_prompt"], result.get("prompt") or "")
        write_text(context.paths["table_interpretation_raw"], result.get("raw_output") or "")
        context.visual_evidence = attach_table_interpretation_to_visual_evidence(
            visual_evidence,
            result,
        )
        context.emit(
            "table_interpretation",
            "done" if result.get("status") in {"ok", "partial"} else result.get("status", "done"),
            (
                "Dedicated table interpretation finished; result attached to visual evidence "
                "for the main LLM parser."
            ),
            interpreter_status=result.get("status"),
            table_count=len(result.get("tables") or []),
            confidence=result.get("overall_confidence"),
            duration_seconds=result.get("duration_seconds"),
            error=(
                "; ".join(result.get("warnings") or [])
                if result.get("status") == "failed"
                else None
            ),
        )

    def _run_table_arbitration(self, context: ExtractionContext) -> None:
        if not context.visual_evidence or not context.preliminary_ocr_context:
            return
        try:
            context.emit(
                "table_arbitration",
                "running",
                "Cross-checking VLM table evidence against OCR layout item/price candidates.",
            )
            context.table_arbitration_result = build_table_arbitration(
                context.visual_evidence,
                context.preliminary_ocr_context,
            )
            save_json(context.paths["table_arbitration"], context.table_arbitration_result)
            context.visual_evidence = attach_table_arbitration_to_visual_evidence(
                context.visual_evidence,
                context.table_arbitration_result,
            )
            context.emit(
                "table_arbitration",
                "done",
                "Table evidence arbitration finished; result attached to visual evidence.",
                summary=context.table_arbitration_result.get("summary"),
                warning_count=len(context.table_arbitration_result.get("warnings") or []),
            )
        except Exception as exc:
            context.table_arbitration_result = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            save_json(context.paths["table_arbitration"], context.table_arbitration_result)
            context.emit(
                "table_arbitration",
                "error",
                "Table evidence arbitration failed; continuing with original visual evidence.",
                error=context.table_arbitration_result["error"],
            )

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
            "VLM-first mode: unloading Ollama/Gemma before VLM to free GPU memory.",
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
        context.emit(
            "gpu_orchestration",
            result.get("status", "done"),
            "Ollama/Gemma unload step finished.",
            mode=result.get("mode"),
            error=result.get("error"),
            duration_seconds=result.get("duration_seconds"),
        )

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
            "VLM-first mode: reloading Ollama/Gemma before the main LLM parser.",
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
        context.emit(
            "gpu_orchestration",
            result.get("status", "done"),
            "Ollama/Gemma reload step finished.",
            mode=result.get("mode"),
            error=result.get("error"),
            duration_seconds=result.get("duration_seconds"),
        )
