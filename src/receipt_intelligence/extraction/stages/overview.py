"""Build the receipt-wide spatial geometry artifacts used by the main parser."""

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
            "Building the receipt-wide spatial document map and same-band row groups.",
        )
        context.spatial_document_map = build_spatial_document_map(
            context.preliminary_ocr_context,
            visual_evidence=context.visual_evidence,
            arbitration=context.table_arbitration_result,
            canvas_width=config.spatial_canvas_width,
        )
        save_json(context.paths["spatial_document_map"], context.spatial_document_map)
        write_text(
            context.paths["spatial_canvas"],
            str(context.spatial_document_map.get("canvas") or ""),
        )

        context.spatial_overview_result = build_geometry_only_overview(context.spatial_document_map)
        save_json(context.paths["spatial_overview"], context.spatial_overview_result)
        context.emit(
            "spatial_overview",
            "done",
            "Receipt-wide geometry map and same-band row groups are ready.",
            geometric_row_group_count=context.spatial_overview_result.get(
                "geometric_row_group_count"
            ),
        )
        return context


__all__ = ["SpatialOverviewStage"]
