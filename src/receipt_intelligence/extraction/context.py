"""Typed workflow state and runtime services for receipt extraction."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.extraction.config import ExtractionConfig
from receipt_intelligence.extraction.contracts.common import StageArtifact
from receipt_intelligence.extraction.dependencies import ExtractionDependencies
from receipt_intelligence.extraction.state import (
    CorrectionArtifacts,
    ExtractionPhase,
    FinalizationArtifacts,
    JsonObject,
    PreparedArtifacts,
    StageContractError,
    StructuredExtractionArtifacts,
    TranscriptionArtifacts,
    ValidationArtifacts,
)
from receipt_intelligence.observability.timing import utc_now_iso

T = TypeVar("T")


def _required(value: T | None, label: str) -> T:
    if value is None:
        raise StageContractError(f"Extraction stage requires {label}.")
    return value


@dataclass(slots=True)
class ExtractionContext:
    """Runtime envelope containing typed artifacts produced by each stage."""

    config: ExtractionConfig
    dependencies: ExtractionDependencies
    started_at: float = field(default_factory=time.perf_counter)
    started_at_utc: str = field(default_factory=utc_now_iso)
    stage_trace: list[JsonObject] = field(default_factory=list)
    logs: list[JsonObject] = field(default_factory=list)
    phase: ExtractionPhase = ExtractionPhase.CREATED
    prepared: PreparedArtifacts | None = None
    transcription: TranscriptionArtifacts | None = None
    structured_extraction: StructuredExtractionArtifacts | None = None
    validation: ValidationArtifacts | None = None
    correction: CorrectionArtifacts | None = None
    finalized: FinalizationArtifacts | None = None

    def emit(self, stage: str, status: str, message: str, **details: Any) -> None:
        event = {
            "stage": stage,
            "status": status,
            "message": message,
            "details": details,
            "source": get_app_version(),
        }
        self.logs.append(event)
        callback = self.config.progress_callback
        if callback is not None:
            try:
                callback(event)
            except Exception:
                pass

    def assert_phase(self, expected: ExtractionPhase, stage_name: str) -> None:
        if self.phase is not expected:
            raise StageContractError(
                f"Stage {stage_name!r} requires phase {expected.value!r}; "
                f"current phase is {self.phase.value!r}."
            )

    def advance_phase(
        self,
        expected: ExtractionPhase,
        target: ExtractionPhase,
        stage_name: str,
    ) -> None:
        self.assert_phase(expected, stage_name)
        self.phase = target

    def begin_transcription_stage(self) -> TranscriptionArtifacts:
        if self.transcription is not None:
            raise StageContractError("Transcription artifacts were already initialized.")
        self.require_prepared()
        self.transcription = TranscriptionArtifacts()
        return self.transcription

    def begin_structured_extraction_stage(self) -> StructuredExtractionArtifacts:
        if self.structured_extraction is not None:
            raise StageContractError("Structured extraction artifacts were already initialized.")
        self.require_transcription()
        self.structured_extraction = StructuredExtractionArtifacts()
        return self.structured_extraction

    def begin_validation_stage(self) -> ValidationArtifacts:
        if self.validation is not None:
            raise StageContractError("Validation artifacts were already initialized.")
        self.require_structured_extraction()
        self.validation = ValidationArtifacts()
        return self.validation

    def begin_correction_stage(self) -> CorrectionArtifacts:
        if self.correction is not None:
            raise StageContractError("Correction artifacts were already initialized.")
        self.require_validation()
        self.correction = CorrectionArtifacts()
        return self.correction

    def register_artifacts(self, artifacts: Iterable[StageArtifact]) -> None:
        paths = self.require_prepared().paths
        for artifact in artifacts:
            paths[artifact.name] = artifact.path

    def require_prepared(self) -> PreparedArtifacts:
        return _required(self.prepared, "prepared artifacts")

    def require_transcription(self) -> TranscriptionArtifacts:
        return _required(self.transcription, "transcription artifacts")

    def require_structured_extraction(self) -> StructuredExtractionArtifacts:
        return _required(self.structured_extraction, "structured extraction artifacts")

    def require_validation(self) -> ValidationArtifacts:
        return _required(self.validation, "validation artifacts")

    def require_correction(self) -> CorrectionArtifacts:
        return _required(self.correction, "correction artifacts")

    def require_finalized(self) -> FinalizationArtifacts:
        return _required(self.finalized, "finalization artifacts")

    @property
    def available_paths(self) -> dict[str, Path]:
        return self.prepared.paths if self.prepared is not None else {}

    @property
    def paths(self) -> dict[str, Path]:
        return self.require_prepared().paths

    @property
    def duration_seconds(self) -> float:
        return round(time.perf_counter() - self.started_at, 2)
