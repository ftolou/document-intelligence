"""Infrastructure composition for the receipt extraction workflow."""

from __future__ import annotations

from receipt_intelligence.adapters.chat import OllamaChatGateway
from receipt_intelligence.adapters.llm import (
    ObservedChatGateway,
    ObservedLlmGateway,
    ObservedMultimodalGateway,
    OllamaGateway,
)
from receipt_intelligence.adapters.multimodal import OllamaMultimodalGateway
from receipt_intelligence.adapters.observability import JsonFileEventSink
from receipt_intelligence.adapters.storage.sqlite.model_calls import SQLiteModelCallRepository
from receipt_intelligence.application.model_call_context import ModelCallContext
from receipt_intelligence.extraction.config import ExtractionConfig
from receipt_intelligence.extraction.correction.artifacts import (
    FilesystemCorrectionArtifactSink,
)
from receipt_intelligence.extraction.correction.composition import (
    build_specialist_correction_service,
)
from receipt_intelligence.extraction.dependencies import ExtractionDependencies
from receipt_intelligence.extraction.settings import PipelineSettings, resolve_transcription_model
from receipt_intelligence.extraction.structured.composition import (
    build_gemma_structured_extraction_service,
)
from receipt_intelligence.extraction.transcription.composition import (
    build_canonical_transcription_service,
    ollama_transcription_options,
)
from receipt_intelligence.extraction.validation.composition import (
    build_deterministic_validation_service,
)
from receipt_intelligence.prompts.registry import default_prompt_registry


def build_extraction_dependencies(config: ExtractionConfig) -> ExtractionDependencies:
    from receipt_intelligence import settings

    grouped = PipelineSettings.from_extraction_config(
        config,
        transcription_model=resolve_transcription_model(),
    )
    extraction_event_sink = JsonFileEventSink(
        config.result_dir / f"{config.run_id}_extraction_metrics.json",
        aliases=(config.result_dir / "latest_extraction_metrics.json",),
    )
    model_call_sink = SQLiteModelCallRepository(
        settings.RECEIPT_DB_PATH,
        enabled=settings.MODEL_CALL_TELEMETRY_ENABLED,
    )
    call_context = ModelCallContext(trace_id=config.run_id, job_id=config.run_id)
    llm_gateway = ObservedLlmGateway(
        OllamaGateway(config.ollama_url),
        model_call_sink,
        default_context=call_context,
    )
    chat_gateway = ObservedChatGateway(
        OllamaChatGateway(config.ollama_url),
        model_call_sink,
        default_context=call_context,
    )
    multimodal_gateway = ObservedMultimodalGateway(
        OllamaMultimodalGateway(
            config.ollama_url,
            generation_options=ollama_transcription_options(grouped),
        ),
        model_call_sink,
        default_context=call_context,
    )
    validation_service = build_deterministic_validation_service(grouped.validation)
    correction_service = build_specialist_correction_service(
        gateway=chat_gateway,
        prompts=default_prompt_registry(),
        validation_service=validation_service,
        settings=grouped.correction,
        artifact_sink=FilesystemCorrectionArtifactSink(
            config.result_dir / f"{config.run_id}_correction"
        ),
    )
    return ExtractionDependencies(
        llm_gateway=llm_gateway,
        event_sink=extraction_event_sink,
        model_call_event_sink=model_call_sink,
        transcription_service=build_canonical_transcription_service(
            grouped,
            multimodal_gateway=multimodal_gateway,
        ),
        structured_extraction_service=build_gemma_structured_extraction_service(
            grouped,
            gateway=chat_gateway,
        ),
        validation_service=validation_service,
        correction_service=correction_service,
    )


__all__ = ["build_extraction_dependencies"]
