"""Typed state artifacts exchanged by extraction workflow stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


class ExtractionPhase(str, Enum):  # noqa: UP042
    """Legal lifecycle positions of the staged extraction workflow."""

    CREATED = "created"
    PREPARED = "prepared"
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
    correction_used: bool = False


@dataclass(slots=True)
class FinalizationArtifacts:
    """Final presentation, categorization, and pipeline metadata artifacts."""

    output_receipt: JsonObject | None = None
    categorization_result: JsonObject | None = None
    categorized_receipt: JsonObject | None = None
    pipeline_meta: JsonObject | None = None
