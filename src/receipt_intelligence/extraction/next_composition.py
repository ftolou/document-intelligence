"""Dependency composition for the opt-in Qwen/Gemma extraction workflow."""

from __future__ import annotations

from dataclasses import replace

from receipt_intelligence.adapters.chat import OllamaChatGateway
from receipt_intelligence.composition import build_extraction_dependencies
from receipt_intelligence.extraction.config import ExtractionConfig
from receipt_intelligence.extraction.correction.artifacts import (
    FilesystemCorrectionArtifactSink,
)
from receipt_intelligence.extraction.correction.composition import (
    build_specialist_correction_service,
)
from receipt_intelligence.extraction.dependencies import ExtractionDependencies
from receipt_intelligence.extraction.settings import PipelineSettings
from receipt_intelligence.extraction.strategy import resolve_transcription_model
from receipt_intelligence.extraction.structured.composition import (
    build_gemma_structured_extraction_service,
)
from receipt_intelligence.extraction.transcription.composition import (
    build_canonical_transcription_service,
)
from receipt_intelligence.extraction.validation.composition import (
    build_deterministic_validation_service,
)
from receipt_intelligence.prompts.registry import default_prompt_registry


def build_next_extraction_dependencies(
    config: ExtractionConfig,
) -> ExtractionDependencies:
    """Extend existing infrastructure dependencies with next-pipeline services."""

    grouped = PipelineSettings.from_extraction_config(
        config,
        transcription_model=resolve_transcription_model(),
    )
    validation_service = build_deterministic_validation_service(grouped.validation)
    correction_service = build_specialist_correction_service(
        gateway=OllamaChatGateway(config.ollama_url),
        prompts=default_prompt_registry(),
        validation_service=validation_service,
        settings=grouped.correction,
        artifact_sink=FilesystemCorrectionArtifactSink(
            config.result_dir / f"{config.run_id}_next_correction"
        ),
    )
    return replace(
        build_extraction_dependencies(config),
        transcription_service=build_canonical_transcription_service(grouped),
        structured_extraction_service=build_gemma_structured_extraction_service(grouped),
        validation_service=validation_service,
        correction_service=correction_service,
    )


__all__ = ["build_next_extraction_dependencies"]
