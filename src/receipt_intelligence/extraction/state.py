"""Typed state artifacts exchanged by the receipt extraction stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from receipt_intelligence.extraction.contracts.correction import CorrectionResult
from receipt_intelligence.extraction.contracts.extraction import StructuredExtractionResult
from receipt_intelligence.extraction.contracts.presentation import (
    CategorizationResult,
    FinalizationResult,
)
from receipt_intelligence.extraction.contracts.transcription import TranscriptionResult
from receipt_intelligence.extraction.contracts.validation import ValidationReport

JsonObject = dict[str, Any]


class ExtractionPhase(str, Enum):  # noqa: UP042
    CREATED = "created"
    PREPARED = "prepared"
    TRANSCRIBED = "transcribed"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    CORRECTED = "corrected"
    CATEGORIZED = "categorized"
    FINALIZED = "finalized"


class StageContractError(RuntimeError):
    """Raised when a stage consumes state that has not been produced yet."""


@dataclass(slots=True)
class PreparedArtifacts:
    paths: dict[str, Path]


@dataclass(slots=True)
class TranscriptionArtifacts:
    result: TranscriptionResult | None = None


@dataclass(slots=True)
class StructuredExtractionArtifacts:
    result: StructuredExtractionResult | None = None


@dataclass(slots=True)
class ValidationArtifacts:
    report: ValidationReport | None = None


@dataclass(slots=True)
class CorrectionArtifacts:
    result: CorrectionResult | None = None


@dataclass(slots=True)
class FinalizationArtifacts:
    categorization: CategorizationResult | None = None
    result: FinalizationResult | None = None
