"""Build the geometry-only receipt-wide spatial evidence artifact."""

from __future__ import annotations

from receipt_intelligence.extraction.artifacts import save_json, write_text
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.evidence.spatial_document import (
    build_spatial_document_map,
)
from receipt_intelligence.extraction.parsing.spatial_overview import (
    build_geometry_only_overview,
)
from receipt_intelligence.extraction.state import ExtractionPhase


class SpatialOverviewStage:
    name = "spatial_overview"
    input_phase = ExtractionPhase.VISUAL_READY
    output_phase = ExtractionPhase.OVERVIEW_READY

    def run(self, context: ExtractionContext) -> ExtractionContext:
        context.begin_overview_stage()
        config = context.config
        if config.extraction_strategy != "spatial_overview":
            context.spatial_overview_result = {
                "schema_version": "spatial_overview_1",
                "status": "skipped",
                "reason": "current_extraction_strategy_selected",
            }
            save_json(context.paths["spatial_overview"], context.spatial_overview_result)
            context.emit(
                "spatial_overview",
                "skipped",
                "Current extraction strategy selected; spatial overview stage was not executed.",
            )
            return context

        if context.preliminary_ocr_context is None:
            context.spatial_overview_result = {
                "schema_version": "spatial_overview_1",
                "status": "failed",
                "warnings": ["Preliminary OCR context was unavailable."],
            }
            save_json(context.paths["spatial_overview"], context.spatial_overview_result)
            context.emit(
                "spatial_overview",
                "error",
                "Spatial overview could not run because OCR geometry was unavailable.",
            )
            return context

        context.emit(
            "spatial_overview",
            "running",
            (
                "Building a receipt-wide spatial document map and geometry-only row groups "
                "without an additional LLM call."
            ),
            strategy=config.extraction_strategy,
        )
        context.spatial_document_map = build_spatial_document_map(
            context.preliminary_ocr_context,
            visual_evidence=context.visual_evidence,
            table_interpretation=context.table_interpretation_result,
            arbitration=context.table_arbitration_result,
            canvas_width=config.spatial_canvas_width,
        )
        save_json(context.paths["spatial_document_map"], context.spatial_document_map)
        write_text(
            context.paths["spatial_canvas"],
            str(context.spatial_document_map.get("canvas") or ""),
        )

        context.spatial_overview_result = build_geometry_only_overview(
            context.spatial_document_map
        )
        result = context.spatial_overview_result
        write_text(context.paths["spatial_overview_prompt"], result.get("prompt") or "")
        write_text(context.paths["spatial_overview_raw"], result.get("raw_output") or "")
        save_json(
            context.paths["spatial_overview"],
            {key: value for key, value in result.items() if key not in {"prompt", "raw_output"}},
        )
        context.emit(
            "spatial_overview",
            "done",
            "Receipt-wide geometry map and same-band row groups are ready.",
            overview_status=result.get("status"),
            llm_call_performed=False,
            geometric_row_group_count=result.get("geometric_row_group_count"),
            warnings=result.get("warnings") or [],
        )
        return context


__all__ = ["SpatialOverviewStage"]
