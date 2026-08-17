"""Prepare one image-first receipt extraction run."""

from __future__ import annotations

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.extraction.artifacts import build_artifact_paths
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.state import ExtractionPhase, PreparedArtifacts


class PreparationStage:
    name = "prepare"
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
                f"{get_app_version()} extraction pipeline started: Paddle geometry -> "
                "Qwen transcription -> Gemma extraction -> read-only validation -> "
                "specialist correction -> optional categorization."
            ),
            workflow="ReceiptExtractionWorkflow",
            source_image_path=str(config.source_image_path),
        )
        return context


__all__ = ["PreparationStage"]
