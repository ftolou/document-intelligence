"""Typed result of Gemma scalar/item extraction and receipt assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

from receipt_intelligence.application.ports.llm import ModelCallMetrics
from receipt_intelligence.extraction.contracts.common import JsonObject, StageArtifact


@dataclass(frozen=True, slots=True)
class StructuredExtractionResult:
    receipt: JsonObject
    scalar_results: tuple[JsonObject, ...] = ()
    item_result: JsonObject | None = None
    diagnostics: JsonObject = field(default_factory=dict)
    model_calls: tuple[ModelCallMetrics, ...] = ()
    artifacts: tuple[StageArtifact, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, dict):
            raise TypeError("StructuredExtractionResult.receipt must be a dictionary.")


__all__ = ["StructuredExtractionResult"]
