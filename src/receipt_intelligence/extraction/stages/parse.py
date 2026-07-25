"""Main LLM parsing and initial deterministic validation."""

from __future__ import annotations

from receipt_intelligence.extraction.artifacts import save_json, write_text
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.evidence.compact import (
    build_compact_evidence,
    compact_evidence_to_prompt_text,
)
from receipt_intelligence.extraction.evidence.grouped import build_grouped_evidence
from receipt_intelligence.extraction.parsing.llm_parser import run_llm_main_parser
from receipt_intelligence.extraction.state import ExtractionPhase
from receipt_intelligence.extraction.validation.consistency import (
    apply_consistency_postprocess,
)
from receipt_intelligence.extraction.validation.receipt import validate_receipt


class MainParsingStage:
    name = "main_parsing"
    input_phase = ExtractionPhase.OVERVIEW_READY
    output_phase = ExtractionPhase.PARSED

    def run(self, context: ExtractionContext) -> ExtractionContext:
        context.begin_parsing_stage()
        config = context.config
        paths = context.paths
        context.emit(
            "llm_main_parser",
            "running",
            (
                "Building VLM-first compact evidence and asking the LLM to extract "
                "the complete receipt JSON."
            ),
            visual_evidence_used=bool(context.visual_evidence),
            model=config.model,
            ollama_url=config.ollama_url,
            num_ctx=config.num_ctx,
            num_predict=config.num_predict,
            max_lines_for_llm=config.max_lines_for_llm,
            json_retry_count=config.json_retry_count,
            format_json=config.format_json,
        )

        main_parser_visual_evidence = context.visual_evidence
        context.llm_result = run_llm_main_parser(
            ocr_json_path=config.ocr_json_path,
            ollama_url=config.ollama_url,
            model=config.model,
            max_lines=config.max_lines_for_llm,
            num_ctx=config.num_ctx,
            num_predict=config.num_predict,
            keep_alive=config.keep_alive,
            timeout=config.llm_timeout_seconds,
            json_retry_count=config.json_retry_count,
            format_json=config.format_json,
            visual_evidence=main_parser_visual_evidence,
            prebuilt_ocr_context=context.preliminary_ocr_context,
            spatial_document_map=context.spatial_document_map,
            llm_gateway=context.dependencies.llm_gateway,
        )

        llm_result = context.llm_result
        context.receipt = llm_result["receipt"]
        context.ocr_context = llm_result["ocr_context"]
        save_json(paths["ocr_context"], context.ocr_context)
        save_json(paths["layout_context"], context.ocr_context.get("layout_context", {}))
        context.compact_evidence = build_compact_evidence(context.ocr_context)
        context.grouped_evidence = build_grouped_evidence(
            context.ocr_context.get("layout_rows") or []
        )
        save_json(paths["compact_evidence"], context.compact_evidence)
        save_json(paths["grouped_evidence"], context.grouped_evidence)
        write_text(
            paths["compact_evidence_text"],
            compact_evidence_to_prompt_text(context.compact_evidence),
        )
        write_text(paths["llm_main_prompt"], llm_result["prompt"])
        write_text(paths["llm_main_raw"], llm_result["raw_output"])
        save_json(paths["receipt_llm_main"], context.receipt)

        self._apply_consistency_postprocess(context)

        if llm_result.get("error"):
            context.emit(
                "llm_main_parser",
                "error",
                "LLM main parser failed. No deterministic fallback was used.",
                error=llm_result.get("error"),
                attempts=llm_result.get("attempts"),
            )
        else:
            context.emit(
                "llm_main_parser",
                "done",
                "LLM main parser returned receipt JSON.",
                item_count=len((context.receipt or {}).get("items") or []),
                attempts=llm_result.get("attempts"),
            )

        context.emit(
            "validation",
            "running",
            "Running deterministic validation only; no semantic fallback or row reconstruction.",
        )
        context.report = validate_receipt(
            context.receipt,
            context.ocr_context,
            tolerance=config.tolerance,
        )
        save_json(paths["validation_report"], context.report)
        context.initial_report = context.report
        context.final_receipt = context.receipt
        context.final_report = context.report
        return context

    def _apply_consistency_postprocess(self, context: ExtractionContext) -> None:
        llm_result = context.llm_result
        if not llm_result.get("error"):
            receipt, actions = apply_consistency_postprocess(
                context.receipt,
                context.visual_evidence,
                context.ocr_context,
                tolerance=max(context.config.tolerance, 0.05),
            )
            context.postprocess_actions = actions
            if actions:
                context.receipt = receipt
                context.emit(
                    "consistency_postprocess",
                    "done",
                    "Applied safe accounting consistency normalizations before validation.",
                    actions=actions,
                )
            save_json(context.paths["receipt_llm_main_postprocessed"], context.receipt)
        save_json(
            context.paths["consistency_postprocess"],
            {"actions": context.postprocess_actions},
        )
