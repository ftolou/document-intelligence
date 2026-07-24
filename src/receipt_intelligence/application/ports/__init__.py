"""Stable application ports implemented by infrastructure adapters."""

from receipt_intelligence.application.ports.events import (
    ApplicationEvent,
    EventSink,
    NullEventSink,
)
from receipt_intelligence.application.ports.jobs import (
    JobDispatcher,
    JobDispatchRequest,
    JobProcessor,
    JobQueueFullError,
    JobRepository,
)
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    GenerationResult,
    GenerationValue,
    LlmGateway,
    ModelCallMetrics,
    coerce_generation_result,
    metrics_to_diagnostics,
)
from receipt_intelligence.application.ports.model_calls import (
    ModelCallFilter,
    ModelCallRepository,
    ModelPricingInput,
)
from receipt_intelligence.application.ports.model_lifecycle import (
    ModelLifecycleCoordinator,
    ModelLifecycleRequest,
    NoOpModelLifecycleCoordinator,
)
from receipt_intelligence.application.ports.ocr import OcrEngine, OcrRequest
from receipt_intelligence.application.ports.receipts import (
    ReceiptEditor,
    ReceiptRepository,
    ReviewApplier,
    ReviewWorkflow,
)
from receipt_intelligence.application.ports.runtime import RuntimeInformation
from receipt_intelligence.application.ports.vlm import VlmEngine, VlmRequest

__all__ = [
    "ApplicationEvent",
    "EventSink",
    "GenerationRequest",
    "GenerationResult",
    "GenerationValue",
    "JobDispatchRequest",
    "JobDispatcher",
    "JobProcessor",
    "JobQueueFullError",
    "JobRepository",
    "LlmGateway",
    "ModelCallFilter",
    "ModelCallMetrics",
    "ModelCallRepository",
    "ModelPricingInput",
    "ModelLifecycleCoordinator",
    "ModelLifecycleRequest",
    "NoOpModelLifecycleCoordinator",
    "NullEventSink",
    "OcrEngine",
    "OcrRequest",
    "ReceiptEditor",
    "ReceiptRepository",
    "ReviewApplier",
    "ReviewWorkflow",
    "RuntimeInformation",
    "VlmEngine",
    "VlmRequest",
    "coerce_generation_result",
    "metrics_to_diagnostics",
]
