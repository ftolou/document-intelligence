"""Injected capabilities required by the extraction workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from receipt_intelligence.application.ports import (
    EventSink,
    LlmGateway,
    NullEventSink,
)
from receipt_intelligence.extraction.services.correction import ReceiptCorrectionService
from receipt_intelligence.extraction.services.structured_extraction import (
    StructuredExtractionService,
)
from receipt_intelligence.extraction.services.transcription import TranscriptionService
from receipt_intelligence.extraction.services.validation import ReceiptValidationService


@dataclass(frozen=True, slots=True)
class ExtractionDependencies:
    llm_gateway: LlmGateway
    event_sink: EventSink
    model_call_event_sink: EventSink = field(default_factory=NullEventSink)
    transcription_service: TranscriptionService | None = None
    structured_extraction_service: StructuredExtractionService | None = None
    validation_service: ReceiptValidationService | None = None
    correction_service: ReceiptCorrectionService | None = None


__all__ = ["ExtractionDependencies"]
