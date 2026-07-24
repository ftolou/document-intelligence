"""Stable application ports implemented by infrastructure adapters."""

from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    GenerationResult,
    GenerationValue,
    LlmGateway,
    ModelCallMetrics,
    coerce_generation_result,
    metrics_to_diagnostics,
)
from receipt_intelligence.application.ports.model_lifecycle import (
    ModelLifecycleCoordinator,
    ModelLifecycleRequest,
    NoOpModelLifecycleCoordinator,
)
from receipt_intelligence.application.ports.ocr import OcrEngine, OcrRequest
from receipt_intelligence.application.ports.vlm import VlmEngine, VlmRequest

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "GenerationValue",
    "LlmGateway",
    "ModelCallMetrics",
    "ModelLifecycleCoordinator",
    "ModelLifecycleRequest",
    "NoOpModelLifecycleCoordinator",
    "OcrEngine",
    "OcrRequest",
    "VlmEngine",
    "VlmRequest",
    "coerce_generation_result",
    "metrics_to_diagnostics",
]
