"""Injected capabilities required by the extraction workflow."""

from __future__ import annotations

from dataclasses import dataclass

from receipt_intelligence.application.ports import (
    EventSink,
    LlmGateway,
    ModelLifecycleCoordinator,
    VlmEngine,
)
from receipt_intelligence.extraction.services.transcription import TranscriptionService


@dataclass(frozen=True, slots=True)
class ExtractionDependencies:
    llm_gateway: LlmGateway
    vlm_engine: VlmEngine
    model_lifecycle: ModelLifecycleCoordinator
    event_sink: EventSink
    transcription_service: TranscriptionService | None = None


__all__ = ["ExtractionDependencies"]
