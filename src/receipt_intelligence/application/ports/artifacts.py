"""Artifact persistence port for extraction stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class ArtifactKind(StrEnum):
    TRANSCRIPTION = "transcription"
    STRUCTURED_EXTRACTION = "structured_extraction"
    INITIAL_RECEIPT = "initial_receipt"
    INITIAL_VALIDATION = "initial_validation"
    CORRECTION_REPORT = "correction_report"
    FINAL_VALIDATION = "final_validation"
    CATEGORIZATION_PROMPT = "categorization_prompt"
    CATEGORIZATION_RAW = "categorization_raw"
    CATEGORIZATION_RESULT = "categorization_result"
    FINAL_RECEIPT = "final_receipt"
    FINAL_RECEIPT_RECONCILED = "final_receipt_reconciled"
    FINAL_RECEIPT_CATEGORIZED = "final_receipt_categorized"
    RECONCILIATION_REPORT = "reconciliation_report"
    PIPELINE_METADATA = "pipeline_metadata"
    STAGE_TRACE = "stage_trace"


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    kind: ArtifactKind
    path: Path
    media_type: str


class ArtifactStore(Protocol):
    """Port for semantic artifact writes and run-directory lifecycle operations."""

    def prepare_run(self, *, run_id: str, overwrite: bool) -> None: ...

    def write_json(
        self,
        *,
        run_id: str,
        kind: ArtifactKind,
        payload: Any,
    ) -> ArtifactReference: ...

    def write_text(
        self,
        *,
        run_id: str,
        kind: ArtifactKind,
        text: str,
    ) -> ArtifactReference: ...

    def publish_aliases(
        self,
        *,
        run_id: str,
        kinds: tuple[ArtifactKind, ...],
    ) -> tuple[ArtifactReference, ...]: ...


__all__ = ["ArtifactKind", "ArtifactReference", "ArtifactStore"]
