"""Public extraction API with lazy exports and no package-import side effects."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from receipt_intelligence.extraction.config import ExtractionConfig, ExtractionRequest
    from receipt_intelligence.extraction.context import ExtractionContext
    from receipt_intelligence.extraction.factory import build_default_extraction_workflow
    from receipt_intelligence.extraction.state import (
        ExtractionPhase,
        FinalizationArtifacts,
        OverviewArtifacts,
        ParsingArtifacts,
        PreparedArtifacts,
        RepairArtifacts,
        StageContractError,
        VisualArtifacts,
    )
    from receipt_intelligence.extraction.workflow import ReceiptExtractionWorkflow

_EXPORTS: dict[str, tuple[str, str]] = {
    "ExtractionConfig": ("receipt_intelligence.extraction.config", "ExtractionConfig"),
    "ExtractionRequest": ("receipt_intelligence.extraction.config", "ExtractionRequest"),
    "ExtractionContext": ("receipt_intelligence.extraction.context", "ExtractionContext"),
    "ExtractionPhase": ("receipt_intelligence.extraction.state", "ExtractionPhase"),
    "StageContractError": ("receipt_intelligence.extraction.state", "StageContractError"),
    "PreparedArtifacts": ("receipt_intelligence.extraction.state", "PreparedArtifacts"),
    "VisualArtifacts": ("receipt_intelligence.extraction.state", "VisualArtifacts"),
    "OverviewArtifacts": ("receipt_intelligence.extraction.state", "OverviewArtifacts"),
    "ParsingArtifacts": ("receipt_intelligence.extraction.state", "ParsingArtifacts"),
    "RepairArtifacts": ("receipt_intelligence.extraction.state", "RepairArtifacts"),
    "FinalizationArtifacts": ("receipt_intelligence.extraction.state", "FinalizationArtifacts"),
    "ReceiptExtractionWorkflow": (
        "receipt_intelligence.extraction.workflow",
        "ReceiptExtractionWorkflow",
    ),
    "build_default_extraction_workflow": (
        "receipt_intelligence.extraction.factory",
        "build_default_extraction_workflow",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve one public extraction symbol only when a caller requests it."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
