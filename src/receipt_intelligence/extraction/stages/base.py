"""Stage protocol for the extraction workflow."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.extraction.context import ExtractionContext


class ExtractionStage(Protocol):
    name: str

    def run(self, context: ExtractionContext) -> ExtractionContext: ...
