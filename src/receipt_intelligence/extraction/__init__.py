"""Public extraction contracts and workflow construction."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ExtractionConfig": ("receipt_intelligence.extraction.config", "ExtractionConfig"),
    "ExtractionRequest": ("receipt_intelligence.extraction.config", "ExtractionRequest"),
    "ExtractionContext": ("receipt_intelligence.extraction.context", "ExtractionContext"),
    "ExtractionPhase": ("receipt_intelligence.extraction.state", "ExtractionPhase"),
    "StageContractError": ("receipt_intelligence.extraction.state", "StageContractError"),
    "PreparedArtifacts": ("receipt_intelligence.extraction.state", "PreparedArtifacts"),
    "FinalizationArtifacts": ("receipt_intelligence.extraction.state", "FinalizationArtifacts"),
    "SourceImageValidationError": (
        "receipt_intelligence.extraction.source_image",
        "SourceImageValidationError",
    ),
    "validate_source_image": (
        "receipt_intelligence.extraction.source_image",
        "validate_source_image",
    ),
    "NormalizedDocumentSource": (
        "receipt_intelligence.extraction.source_normalization",
        "NormalizedDocumentSource",
    ),
    "SourceNormalizationError": (
        "receipt_intelligence.extraction.source_normalization",
        "SourceNormalizationError",
    ),
    "SourceNormalizationLimits": (
        "receipt_intelligence.extraction.source_normalization",
        "SourceNormalizationLimits",
    ),
    "VisualPage": (
        "receipt_intelligence.extraction.source_normalization",
        "VisualPage",
    ),
    "normalize_document_source": (
        "receipt_intelligence.extraction.source_normalization",
        "normalize_document_source",
    ),
    "ReceiptExtractionWorkflow": (
        "receipt_intelligence.extraction.workflow",
        "ReceiptExtractionWorkflow",
    ),
    "build_extraction_workflow": (
        "receipt_intelligence.extraction.factory",
        "build_extraction_workflow",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
