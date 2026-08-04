"""Application composition helpers selecting concrete infrastructure adapters."""

from __future__ import annotations

from receipt_intelligence.adapters.jobs import ThreadPoolJobDispatcher
from receipt_intelligence.adapters.lifecycle import OllamaModelLifecycleCoordinator
from receipt_intelligence.adapters.llm import ObservedLlmGateway, OllamaGateway
from receipt_intelligence.adapters.observability import JsonFileEventSink
from receipt_intelligence.adapters.ocr import PaddleOcrEngine
from receipt_intelligence.adapters.storage.sqlite.model_calls import (
    SQLiteModelCallRepository,
)
from receipt_intelligence.application.model_call_context import ModelCallContext
from receipt_intelligence.application.ports import (
    JobDispatcher,
    JobProcessor,
    JobRepository,
    OcrEngine,
)
from receipt_intelligence.extraction.config import ExtractionConfig
from receipt_intelligence.extraction.dependencies import ExtractionDependencies
from receipt_intelligence.vlm_client_composition import build_client_vlm_engine


def build_extraction_dependencies(config: ExtractionConfig) -> ExtractionDependencies:
    from receipt_intelligence import settings

    extraction_event_sink = JsonFileEventSink(
        config.result_dir / f"{config.run_id}_extraction_metrics.json",
        aliases=(config.result_dir / "latest_extraction_metrics.json",),
    )
    model_call_sink = SQLiteModelCallRepository(
        settings.RECEIPT_DB_PATH, enabled=settings.MODEL_CALL_TELEMETRY_ENABLED
    )
    return ExtractionDependencies(
        llm_gateway=ObservedLlmGateway(
            OllamaGateway(config.ollama_url),
            model_call_sink,
            default_context=ModelCallContext(
                trace_id=config.run_id,
                job_id=config.run_id,
            ),
        ),
        vlm_engine=build_client_vlm_engine(config),
        model_lifecycle=OllamaModelLifecycleCoordinator(
            base_url=config.ollama_url,
            control_mode=config.ollama_control_mode,
            unload_command=config.ollama_unload_command,
            start_command=config.ollama_start_command,
        ),
        event_sink=extraction_event_sink,
        model_call_event_sink=model_call_sink,
    )


def build_job_dispatcher(
    repository: JobRepository,
    processor: JobProcessor,
    *,
    max_workers: int = 1,
    queue_capacity: int = 32,
    claim_lease_seconds: float = 120.0,
    maintenance_interval_seconds: float = 10.0,
) -> JobDispatcher:
    return ThreadPoolJobDispatcher(
        repository,
        processor,
        max_workers=max_workers,
        queue_capacity=queue_capacity,
        claim_lease_seconds=claim_lease_seconds,
        maintenance_interval_seconds=maintenance_interval_seconds,
    )


def build_ocr_engine() -> OcrEngine:
    return PaddleOcrEngine()


__all__ = [
    "build_client_vlm_engine",
    "build_extraction_dependencies",
    "build_job_dispatcher",
    "build_ocr_engine",
]
