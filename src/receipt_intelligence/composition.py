"""Application composition helpers selecting concrete infrastructure adapters."""

from __future__ import annotations

import os

from receipt_intelligence.adapters.jobs import ThreadPoolJobDispatcher
from receipt_intelligence.adapters.lifecycle import OllamaModelLifecycleCoordinator
from receipt_intelligence.adapters.llm import OllamaGateway
from receipt_intelligence.adapters.ocr import PaddleOcrEngine
from receipt_intelligence.adapters.vlm import (
    PaddleCliVlmEngine,
    PaddlePythonVlmEngine,
    RemoteVlmClient,
    TrustedCommandVlmEngine,
)
from receipt_intelligence.application.ports import (
    JobDispatcher,
    JobProcessor,
    JobRepository,
    OcrEngine,
    VlmEngine,
)
from receipt_intelligence.application.vlm import (
    FallbackVlmEngine,
    OptionalVlmEngine,
    UnsupportedVlmEngine,
)
from receipt_intelligence.extraction.config import ExtractionConfig
from receipt_intelligence.extraction.dependencies import ExtractionDependencies

_REMOTE_BACKENDS = {"http_service", "http-service", "service", "vlm_service"}
_LOCAL_BACKENDS = {"paddleocr_vl", "paddleocr-vl", "paddleocrvl", "local"}
_CLI_RUNNERS = {"cli", "doc_parser", "paddleocr_cli"}
_PYTHON_RUNNERS = {"python", "python_api"}


def _local_vlm_engine(runner_name: str) -> VlmEngine:
    runner = (runner_name or "auto").strip().lower()
    if runner in _CLI_RUNNERS:
        return PaddleCliVlmEngine()
    if runner in _PYTHON_RUNNERS:
        return PaddlePythonVlmEngine()
    return FallbackVlmEngine(PaddlePythonVlmEngine(), PaddleCliVlmEngine())


def build_client_vlm_engine(config: ExtractionConfig) -> VlmEngine:
    """Build the VLM capability used by the receipt application."""
    requested_backend = (config.vlm_backend or "http_service").strip().lower()
    if config.vlm_command.strip():
        delegate: VlmEngine = TrustedCommandVlmEngine(config.vlm_command)
        effective_backend = "command"
    elif requested_backend in _REMOTE_BACKENDS:
        delegate = RemoteVlmClient(config.vlm_service_url)
        effective_backend = "http_service"
    elif requested_backend in _LOCAL_BACKENDS:
        allow_local = os.getenv("VLM_ALLOW_LOCAL_BACKEND", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if allow_local:
            runner = os.getenv("VLM_RUNNER", os.getenv("VLM_SERVICE_RUNNER", "cli"))
            delegate = _local_vlm_engine(runner)
            effective_backend = requested_backend
        else:
            delegate = RemoteVlmClient(config.vlm_service_url)
            effective_backend = "http_service"
    else:
        delegate = UnsupportedVlmEngine(requested_backend)
        effective_backend = requested_backend

    return OptionalVlmEngine(delegate, backend_name=effective_backend)


def build_vlm_service_engine(
    *,
    backend_name: str,
    runner_name: str,
    trusted_command: str = "",
) -> VlmEngine:
    """Build the local engine hosted by the standalone VLM HTTP service."""
    if trusted_command.strip():
        return TrustedCommandVlmEngine(trusted_command)
    backend = (backend_name or "paddleocr_vl").strip().lower()
    if backend in _LOCAL_BACKENDS:
        return _local_vlm_engine(runner_name)
    return UnsupportedVlmEngine(backend)


def build_extraction_dependencies(config: ExtractionConfig) -> ExtractionDependencies:
    return ExtractionDependencies(
        llm_gateway=OllamaGateway(config.ollama_url),
        vlm_engine=build_client_vlm_engine(config),
        model_lifecycle=OllamaModelLifecycleCoordinator(
            base_url=config.ollama_url,
            control_mode=config.ollama_control_mode,
            unload_command=config.ollama_unload_command,
            start_command=config.ollama_start_command,
        ),
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
    "build_vlm_service_engine",
]
