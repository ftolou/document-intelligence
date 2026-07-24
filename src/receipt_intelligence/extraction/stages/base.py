"""Stage protocol for the extraction workflow."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.state import ExtractionPhase


class ExtractionStage(Protocol):
    name: str
    input_phase: ExtractionPhase
    output_phase: ExtractionPhase

    def run(self, context: ExtractionContext) -> ExtractionContext: ...
