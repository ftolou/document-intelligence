"""Typed state artifacts exchanged by extraction workflow stages."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """Legal lifecycle positions of the staged extraction workflow."""

    CREATED = "created"
    PREPARED = "prepared"
    TRANSCRIBED = "transcribed"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    CORRECTED = "corrected"
    CATEGORIZED = "categorized"
    VISUAL_READY = "visual_ready"
    OVERVIEW_READY = "overview_ready"
    PARSED = "parsed"
    REPAIRED = "repaired"
    FINALIZED = "finalized"


class StageContractError(RuntimeError):
    """Raised when a stage consumes state that has not been produced yet."""


@dataclass(slots=True)
class PreparedArtifacts:
    """Artifacts produced while preparing one extraction run."""

    paths: dict[str, Path]
    preliminary_ocr_context: JsonObject | None = None


@dataclass(slots=True)
class TranscriptionArtifacts:
    """Canonical Paddle/Qwen evidence produced by the next transcription stage."""

    result: TranscriptionResult | None = None


@dataclass(slots=True)
class StructuredExtractionArtifacts:
    """Gemma scalar/item output produced by the next extraction stage."""

    result: StructuredExtractionResult | None = None


@dataclass(slots=True)
class ValidationArtifacts:
    """Read-only deterministic report for the next extraction path."""

    report: ValidationReport | None = None


@dataclass(slots=True)
class CorrectionArtifacts:
    """Specialist correction result selected by deterministic acceptance."""

    result: CorrectionResult | None = None


@dataclass(slots=True)
class VisualArtifacts:
    """VLM, crop OCR, and table evidence produced before main parsing."""

    visual_result: JsonObject | None = None
    visual_evidence: JsonObject | None = None
    region_reocr_result: JsonObject | None = None
    table_arbitration_result: JsonObject | None = None


@dataclass(slots=True)
class OverviewArtifacts:
    """Canonical spatial map and geometry-only overview metadata."""

    spatial_document_map: JsonObject | None = None
    spatial_overview_result: JsonObject | None = None


@dataclass(slots=True)
class ParsingArtifacts:
    """Main parser output and deterministic validation state."""

    llm_result: JsonObject | None = None
    receipt: JsonObject | None = None
    ocr_context: JsonObject | None = None
    compact_evidence: JsonObject | None = None
    grouped_evidence: JsonObject | None = None
    postprocess_actions: list[JsonObject] = field(default_factory=list)
    initial_report: JsonObject | None = None
    current_report: JsonObject | None = None


@dataclass(slots=True)
class RepairArtifacts:
    """Recovery attempts and the currently selected receipt candidate."""

    selected_receipt: JsonObject
    selected_report: JsonObject
    reocr_result: JsonObject | None = None
    patch_correction_result: JsonObject | None = None
    corrected_report: JsonObject | None = None
    semantic_suspicion_result: JsonObject | None = None
    corrected_semantic_suspicion_result: JsonObject | None = None
    correction_used: bool = False


@dataclass(slots=True)
class FinalizationArtifacts:
    """Final presentation, categorization, and pipeline metadata artifacts."""

    output_receipt: JsonObject | None = None
    categorization_result: JsonObject | None = None
    categorized_receipt: JsonObject | None = None
    pipeline_meta: JsonObject | None = None
    next_categorization: CategorizationResult | None = None
    next_finalization: FinalizationResult | None = None
