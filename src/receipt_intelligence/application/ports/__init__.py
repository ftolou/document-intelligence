"""Stable application ports implemented by infrastructure adapters.

Exports are resolved lazily so a service imports only the port modules it uses.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from receipt_intelligence.application.ports.artifacts import (
        ArtifactKind,
        ArtifactReference,
        ArtifactStore,
    )
    from receipt_intelligence.application.ports.embeddings import (
        CloseableEmbeddingGateway,
        EmbeddingBatchResult,
        EmbeddingGateway,
        EmbeddingProviderError,
    )
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
    from receipt_intelligence.application.ports.multimodal import (
        MultimodalGateway,
        MultimodalGenerationRequest,
        MultimodalGenerationResult,
    )
    from receipt_intelligence.application.ports.receipts import (
        ReceiptEditor,
        ReceiptRepository,
        ReviewApplier,
        ReviewWorkflow,
    )
    from receipt_intelligence.application.ports.runtime import RuntimeInformation
    from receipt_intelligence.application.ports.text_detection import (
        DetectedTextRegion,
        Point,
        Polygon,
        TextDetectionEngine,
        TextDetectionRequest,
        TextDetectionResult,
    )

    _TYPE_EXPORTS = (
        ArtifactKind,
        ArtifactReference,
        ArtifactStore,
        CloseableEmbeddingGateway,
        EmbeddingBatchResult,
        EmbeddingGateway,
        EmbeddingProviderError,
        ApplicationEvent,
        EventSink,
        NullEventSink,
        JobDispatcher,
        JobDispatchRequest,
        JobProcessor,
        JobQueueFullError,
        JobRepository,
        GenerationRequest,
        GenerationResult,
        GenerationValue,
        LlmGateway,
        ModelCallMetrics,
        coerce_generation_result,
        metrics_to_diagnostics,
        MultimodalGateway,
        MultimodalGenerationRequest,
        MultimodalGenerationResult,
        ModelCallFilter,
        ModelCallRepository,
        ModelPricingInput,
        ReceiptEditor,
        ReceiptRepository,
        ReviewApplier,
        ReviewWorkflow,
        RuntimeInformation,
        DetectedTextRegion,
        Point,
        Polygon,
        TextDetectionEngine,
        TextDetectionRequest,
        TextDetectionResult,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "ArtifactKind": ("artifacts", "ArtifactKind"),
    "ArtifactReference": ("artifacts", "ArtifactReference"),
    "ArtifactStore": ("artifacts", "ArtifactStore"),
    "CloseableEmbeddingGateway": ("embeddings", "CloseableEmbeddingGateway"),
    "EmbeddingBatchResult": ("embeddings", "EmbeddingBatchResult"),
    "EmbeddingGateway": ("embeddings", "EmbeddingGateway"),
    "EmbeddingProviderError": ("embeddings", "EmbeddingProviderError"),
    "ApplicationEvent": ("events", "ApplicationEvent"),
    "EventSink": ("events", "EventSink"),
    "NullEventSink": ("events", "NullEventSink"),
    "JobDispatcher": ("jobs", "JobDispatcher"),
    "JobDispatchRequest": ("jobs", "JobDispatchRequest"),
    "JobProcessor": ("jobs", "JobProcessor"),
    "JobQueueFullError": ("jobs", "JobQueueFullError"),
    "JobRepository": ("jobs", "JobRepository"),
    "GenerationRequest": ("llm", "GenerationRequest"),
    "GenerationResult": ("llm", "GenerationResult"),
    "GenerationValue": ("llm", "GenerationValue"),
    "LlmGateway": ("llm", "LlmGateway"),
    "ModelCallMetrics": ("llm", "ModelCallMetrics"),
    "coerce_generation_result": ("llm", "coerce_generation_result"),
    "metrics_to_diagnostics": ("llm", "metrics_to_diagnostics"),
    "MultimodalGateway": ("multimodal", "MultimodalGateway"),
    "MultimodalGenerationRequest": ("multimodal", "MultimodalGenerationRequest"),
    "MultimodalGenerationResult": ("multimodal", "MultimodalGenerationResult"),
    "ModelCallFilter": ("model_calls", "ModelCallFilter"),
    "ModelCallRepository": ("model_calls", "ModelCallRepository"),
    "ModelPricingInput": ("model_calls", "ModelPricingInput"),
    "ReceiptEditor": ("receipts", "ReceiptEditor"),
    "ReceiptRepository": ("receipts", "ReceiptRepository"),
    "ReviewApplier": ("receipts", "ReviewApplier"),
    "ReviewWorkflow": ("receipts", "ReviewWorkflow"),
    "RuntimeInformation": ("runtime", "RuntimeInformation"),
    "DetectedTextRegion": ("text_detection", "DetectedTextRegion"),
    "Point": ("text_detection", "Point"),
    "Polygon": ("text_detection", "Polygon"),
    "TextDetectionEngine": ("text_detection", "TextDetectionEngine"),
    "TextDetectionRequest": ("text_detection", "TextDetectionRequest"),
    "TextDetectionResult": ("text_detection", "TextDetectionResult"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
