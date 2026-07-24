"""Injected capabilities required by the extraction workflow."""

from __future__ import annotations

from dataclasses import dataclass

from receipt_intelligence.application.ports import (
    LlmGateway,
    ModelLifecycleCoordinator,
    VlmEngine,
)


@dataclass(frozen=True, slots=True)
class ExtractionDependencies:
    llm_gateway: LlmGateway
    vlm_engine: VlmEngine
    model_lifecycle: ModelLifecycleCoordinator


__all__ = ["ExtractionDependencies"]
