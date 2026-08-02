"""Application boundary for canonical receipt transcription."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.contracts.transcription import (
    TranscriptionRequest,
    TranscriptionResult,
)


class TranscriptionService(Protocol):
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...


__all__ = ["TranscriptionService"]
