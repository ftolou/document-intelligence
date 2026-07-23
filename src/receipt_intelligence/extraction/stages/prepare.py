"""Prepare artifacts and preliminary OCR context."""

from __future__ import annotations

import json

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.extraction.artifacts import build_artifact_paths
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.parsing.llm_parser import build_ocr_context


class PreparationStage:
    name = "prepare"

    def run(self, context: ExtractionContext) -> ExtractionContext:
        config = context.config
        config.result_dir.mkdir(parents=True, exist_ok=True)
        context.paths = build_artifact_paths(config.result_dir, config.run_id)

        context.emit(
            "pipeline",
            "running",
            (
                f"{get_app_version()} staged pipeline started: VLM layout regions -> "
                "crop re-OCR -> compact LLM parse -> validation -> optional "
                "patch-only targeted correction -> optional item categorization."
            ),
            workflow="ReceiptExtractionWorkflow",
        )

        try:
            raw_ocr = json.loads(config.ocr_json_path.read_text(encoding="utf-8-sig"))
            context.preliminary_ocr_context = build_ocr_context(
                raw_ocr,
                max_lines=config.max_lines_for_llm,
            )
        except Exception as exc:
            context.preliminary_ocr_context = None
            context.emit(
                "table_arbitration",
                "error",
                (
                    "Could not build preliminary OCR context for table arbitration; "
                    "continuing without it."
                ),
                error=f"{type(exc).__name__}: {exc}",
            )
        return context
