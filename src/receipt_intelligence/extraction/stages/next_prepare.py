"""Preparation stage for the image-first Qwen/Gemma workflow."""

from __future__ import annotations

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.extraction.artifacts import build_artifact_paths
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.state import ExtractionPhase, PreparedArtifacts


class NextPreparationStage:
    name = "next_prepare"
    input_phase = ExtractionPhase.CREATED
    output_phase = ExtractionPhase.PREPARED

    def run(self, context: ExtractionContext) -> ExtractionContext:
        config = context.config
        config.result_dir.mkdir(parents=True, exist_ok=True)
        context.prepared = PreparedArtifacts(
            paths=build_artifact_paths(config.result_dir, config.run_id)
        )
        context.emit(
            "pipeline",
            "running",
            (
                f"{get_app_version()} next extraction pipeline started: Paddle geometry -> "
                "Qwen transcription -> Gemma extraction -> read-only validation -> "
                "specialist correction -> optional categorization."
            ),
            workflow="NextReceiptExtractionWorkflow",
            source_image_path=(str(config.source_image_path) if config.source_image_path else None),
            legacy_ocr_available=bool(config.ocr_json_path),
        )
        return context


__all__ = ["NextPreparationStage"]
