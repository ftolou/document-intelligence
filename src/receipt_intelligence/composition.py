"""Application composition helpers selecting concrete infrastructure adapters."""

from __future__ import annotations

from receipt_intelligence.adapters.lifecycle import OllamaModelLifecycleCoordinator
from receipt_intelligence.adapters.llm import OllamaGateway
from receipt_intelligence.adapters.ocr import PaddleOcrEngine
from receipt_intelligence.adapters.vlm import ConfiguredVlmEngine
from receipt_intelligence.application.ports import OcrEngine
from receipt_intelligence.extraction.config import ExtractionConfig
from receipt_intelligence.extraction.dependencies import ExtractionDependencies


def build_extraction_dependencies(config: ExtractionConfig) -> ExtractionDependencies:
    return ExtractionDependencies(
        llm_gateway=OllamaGateway(config.ollama_url),
        vlm_engine=ConfiguredVlmEngine(
            backend_name=config.vlm_backend,
            service_url=config.vlm_service_url,
            trusted_command=config.vlm_command,
        ),
        model_lifecycle=OllamaModelLifecycleCoordinator(
            base_url=config.ollama_url,
            control_mode=config.ollama_control_mode,
            unload_command=config.ollama_unload_command,
            start_command=config.ollama_start_command,
        ),
    )


def build_ocr_engine() -> OcrEngine:
    return PaddleOcrEngine()


__all__ = ["build_extraction_dependencies", "build_ocr_engine"]
