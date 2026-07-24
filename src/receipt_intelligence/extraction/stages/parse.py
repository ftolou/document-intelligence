"""Main LLM parsing, table assembly and initial deterministic validation."""

from __future__ import annotations

from receipt_intelligence.extraction.artifacts import save_json, write_text
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.evidence.compact import (
    build_compact_evidence,
    compact_evidence_to_prompt_text,
)
from receipt_intelligence.extraction.evidence.grouped import build_grouped_evidence
from receipt_intelligence.extraction.parsing.llm_parser import run_llm_main_parser
from receipt_intelligence.extraction.parsing.table_assembler import (
    assemble_receipt_from_table_interpretation,
    compact_visual_evidence_for_main_parser,
    merge_authoritative_table_items,
)
from receipt_intelligence.extraction.validation.consistency import (
    apply_consistency_postprocess,
)
from receipt_intelligence.extraction.validation.receipt import validate_receipt


class MainParsingStage:
    name = "main_parsing"

    def run(self, context: ExtractionContext) -> ExtractionContext:
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

        main_parser_visual_evidence = (
            compact_visual_evidence_for_main_parser(
                context.visual_evidence,
                table_interpretation=context.table_interpretation_result,
                arbitration=context.table_arbitration_result,
            )
            if context.visual_evidence
            else None
        )
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

        self._apply_table_assembly(context)
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
            context.require("receipt"),
            context.require("ocr_context"),
            tolerance=config.tolerance,
        )
        save_json(paths["validation_report"], context.report)
        context.initial_report = context.report
        context.final_receipt = context.receipt
        context.final_report = context.report
        return context

    def _apply_table_assembly(self, context: ExtractionContext) -> None:
        if context.table_interpretation_result is None:
            return
        llm_result = context.require("llm_result")
        receipt = context.require("receipt")
        if llm_result.get("error"):
            recovered_error = llm_result.get("error")
            receipt = assemble_receipt_from_table_interpretation(
                table_interpretation=context.table_interpretation_result,
                arbitration=context.table_arbitration_result,
                base_receipt=receipt,
                reason="main_parser_failed_recovered_from_llm_table_interpretation",
            )
            llm_result["recovered_error"] = recovered_error
            llm_result["error"] = None
            context.table_assembly_report = {
                "attempted": True,
                "changed": True,
                "mode": "assembled_after_main_parser_failure",
                "recovered_error": recovered_error,
            }
            context.emit(
                "table_assembly",
                "done",
                (
                    "Main parser failed; assembled a provisional receipt from dedicated "
                    "LLM table interpretation."
                ),
                recovered_error=recovered_error,
                item_count=len(receipt.get("items") or []),
            )
        else:
            receipt, report = merge_authoritative_table_items(
                receipt,
                context.table_interpretation_result,
                context.table_arbitration_result,
            )
            report["attempted"] = True
            context.table_assembly_report = report
            if report.get("changed"):
                context.emit(
                    "table_assembly",
                    "done",
                    (
                        "Replaced weak main-parser item rows with authoritative LLM table "
                        "interpretation/arbitration rows."
                    ),
                    item_count=len(receipt.get("items") or []),
                    source=report.get("source"),
                )
        context.receipt = receipt
        save_json(context.paths["receipt_table_assembled"], receipt)
        save_json(context.paths["table_assembly_report"], context.table_assembly_report)

    def _apply_consistency_postprocess(self, context: ExtractionContext) -> None:
        llm_result = context.require("llm_result")
        if not llm_result.get("error"):
            receipt, actions = apply_consistency_postprocess(
                context.require("receipt"),
                context.visual_evidence,
                context.require("ocr_context"),
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
