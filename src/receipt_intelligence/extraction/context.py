"""Typed workflow state and runtime services for receipt extraction."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.extraction.config import ExtractionConfig
from receipt_intelligence.extraction.dependencies import ExtractionDependencies
from receipt_intelligence.extraction.state import (
    ExtractionPhase,
    FinalizationArtifacts,
    JsonObject,
    OverviewArtifacts,
    ParsingArtifacts,
    PreparedArtifacts,
    RepairArtifacts,
    StageContractError,
    StructuredExtractionArtifacts,
    TranscriptionArtifacts,
    ValidationArtifacts,
    VisualArtifacts,
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
    visual: VisualArtifacts | None = None
    overview: OverviewArtifacts | None = None
    parsed: ParsingArtifacts | None = None
    repair: RepairArtifacts | None = None
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
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            # Progress reporting is best effort and must never fail extraction.
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

    def begin_visual_stage(self) -> VisualArtifacts:
        if self.visual is not None:
            raise StageContractError("Visual artifacts were already initialized.")
        self.require_prepared()
        self.visual = VisualArtifacts()
        return self.visual

    def begin_overview_stage(self) -> OverviewArtifacts:
        if self.overview is not None:
            raise StageContractError("Overview artifacts were already initialized.")
        self.require_visual()
        self.overview = OverviewArtifacts()
        return self.overview

    def begin_parsing_stage(self) -> ParsingArtifacts:
        if self.parsed is not None:
            raise StageContractError("Parsing artifacts were already initialized.")
        self.require_overview()
        self.parsed = ParsingArtifacts()
        return self.parsed

    def begin_repair_stage(self) -> RepairArtifacts:
        if self.repair is not None:
            raise StageContractError("Repair artifacts were already initialized.")
        parsed = self.require_parsed()
        self.repair = RepairArtifacts(
            selected_receipt=_required(parsed.receipt, "parsed receipt"),
            selected_report=_required(parsed.current_report, "initial validation report"),
        )
        return self.repair

    def begin_finalization_stage(self) -> FinalizationArtifacts:
        if self.finalized is not None:
            raise StageContractError("Finalization artifacts were already initialized.")
        self.require_repair()
        self.finalized = FinalizationArtifacts()
        return self.finalized

    def require_prepared(self) -> PreparedArtifacts:
        return _required(self.prepared, "prepared artifacts")

    def require_transcription(self) -> TranscriptionArtifacts:
        return _required(self.transcription, "transcription artifacts")

    def require_structured_extraction(self) -> StructuredExtractionArtifacts:
        return _required(self.structured_extraction, "structured extraction artifacts")

    def require_validation(self) -> ValidationArtifacts:
        return _required(self.validation, "validation artifacts")

    def require_visual(self) -> VisualArtifacts:
        return _required(self.visual, "visual artifacts")

    def require_overview(self) -> OverviewArtifacts:
        return _required(self.overview, "overview artifacts")

    def require_parsed(self) -> ParsingArtifacts:
        return _required(self.parsed, "parsing artifacts")

    def require_repair(self) -> RepairArtifacts:
        return _required(self.repair, "repair artifacts")

    def require_finalized(self) -> FinalizationArtifacts:
        return _required(self.finalized, "finalization artifacts")

    @property
    def available_paths(self) -> dict[str, Path]:
        return self.prepared.paths if self.prepared is not None else {}

    @property
    def paths(self) -> dict[str, Path]:
        return self.require_prepared().paths

    @property
    def preliminary_ocr_context(self) -> JsonObject | None:
        return self.require_prepared().preliminary_ocr_context

    @preliminary_ocr_context.setter
    def preliminary_ocr_context(self, value: JsonObject | None) -> None:
        self.require_prepared().preliminary_ocr_context = value

    @property
    def visual_result(self) -> JsonObject | None:
        return self.visual.visual_result if self.visual is not None else None

    @visual_result.setter
    def visual_result(self, value: JsonObject | None) -> None:
        self.require_visual().visual_result = value

    @property
    def visual_evidence(self) -> JsonObject | None:
        return self.visual.visual_evidence if self.visual is not None else None

    @visual_evidence.setter
    def visual_evidence(self, value: JsonObject | None) -> None:
        self.require_visual().visual_evidence = value

    @property
    def region_reocr_result(self) -> JsonObject | None:
        return self.visual.region_reocr_result if self.visual is not None else None

    @region_reocr_result.setter
    def region_reocr_result(self, value: JsonObject | None) -> None:
        self.require_visual().region_reocr_result = value

    @property
    def table_arbitration_result(self) -> JsonObject | None:
        return self.visual.table_arbitration_result if self.visual is not None else None

    @table_arbitration_result.setter
    def table_arbitration_result(self, value: JsonObject | None) -> None:
        self.require_visual().table_arbitration_result = value

    @property
    def spatial_document_map(self) -> JsonObject | None:
        return self.overview.spatial_document_map if self.overview is not None else None

    @spatial_document_map.setter
    def spatial_document_map(self, value: JsonObject | None) -> None:
        self.require_overview().spatial_document_map = value

    @property
    def spatial_overview_result(self) -> JsonObject | None:
        return self.overview.spatial_overview_result if self.overview is not None else None

    @spatial_overview_result.setter
    def spatial_overview_result(self, value: JsonObject | None) -> None:
        self.require_overview().spatial_overview_result = value

    @property
    def llm_result(self) -> JsonObject:
        return _required(self.require_parsed().llm_result, "main parser result")

    @llm_result.setter
    def llm_result(self, value: JsonObject) -> None:
        self.require_parsed().llm_result = value

    @property
    def receipt(self) -> JsonObject:
        if self.repair is not None:
            return self.repair.selected_receipt
        return _required(self.require_parsed().receipt, "parsed receipt")

    @receipt.setter
    def receipt(self, value: JsonObject) -> None:
        if self.repair is not None:
            self.repair.selected_receipt = value
        else:
            self.require_parsed().receipt = value

    @property
    def ocr_context(self) -> JsonObject:
        return _required(self.require_parsed().ocr_context, "OCR context")

    @ocr_context.setter
    def ocr_context(self, value: JsonObject) -> None:
        self.require_parsed().ocr_context = value

    @property
    def compact_evidence(self) -> JsonObject:
        return _required(self.require_parsed().compact_evidence, "compact evidence")

    @compact_evidence.setter
    def compact_evidence(self, value: JsonObject) -> None:
        self.require_parsed().compact_evidence = value

    @property
    def grouped_evidence(self) -> JsonObject:
        return _required(self.require_parsed().grouped_evidence, "grouped evidence")

    @grouped_evidence.setter
    def grouped_evidence(self, value: JsonObject) -> None:
        self.require_parsed().grouped_evidence = value

    @property
    def postprocess_actions(self) -> list[JsonObject]:
        return self.require_parsed().postprocess_actions

    @postprocess_actions.setter
    def postprocess_actions(self, value: list[JsonObject]) -> None:
        self.require_parsed().postprocess_actions = value

    @property
    def initial_report(self) -> JsonObject:
        return _required(self.require_parsed().initial_report, "initial validation report")

    @initial_report.setter
    def initial_report(self, value: JsonObject) -> None:
        self.require_parsed().initial_report = value

    @property
    def report(self) -> JsonObject:
        if self.repair is not None:
            return self.repair.selected_report
        return _required(self.require_parsed().current_report, "current validation report")

    @report.setter
    def report(self, value: JsonObject) -> None:
        if self.repair is not None:
            self.repair.selected_report = value
        else:
            self.require_parsed().current_report = value

    @property
    def final_receipt(self) -> JsonObject:
        return self.receipt

    @final_receipt.setter
    def final_receipt(self, value: JsonObject) -> None:
        self.receipt = value

    @property
    def final_report(self) -> JsonObject:
        return self.report

    @final_report.setter
    def final_report(self, value: JsonObject) -> None:
        self.report = value

    @property
    def reocr_result(self) -> JsonObject | None:
        return self.repair.reocr_result if self.repair is not None else None

    @reocr_result.setter
    def reocr_result(self, value: JsonObject | None) -> None:
        self.require_repair().reocr_result = value

    @property
    def patch_correction_result(self) -> JsonObject | None:
        return self.repair.patch_correction_result if self.repair is not None else None

    @patch_correction_result.setter
    def patch_correction_result(self, value: JsonObject | None) -> None:
        self.require_repair().patch_correction_result = value

    @property
    def corrected_report(self) -> JsonObject | None:
        return self.repair.corrected_report if self.repair is not None else None

    @corrected_report.setter
    def corrected_report(self, value: JsonObject | None) -> None:
        self.require_repair().corrected_report = value

    @property
    def semantic_suspicion_result(self) -> JsonObject | None:
        return self.repair.semantic_suspicion_result if self.repair is not None else None

    @semantic_suspicion_result.setter
    def semantic_suspicion_result(self, value: JsonObject | None) -> None:
        self.require_repair().semantic_suspicion_result = value

    @property
    def corrected_semantic_suspicion_result(self) -> JsonObject | None:
        return self.repair.corrected_semantic_suspicion_result if self.repair is not None else None

    @corrected_semantic_suspicion_result.setter
    def corrected_semantic_suspicion_result(self, value: JsonObject | None) -> None:
        self.require_repair().corrected_semantic_suspicion_result = value

    @property
    def semantic_suspicion_result(self) -> JsonObject | None:
        return self.repair.semantic_suspicion_result if self.repair is not None else None

    @semantic_suspicion_result.setter
    def semantic_suspicion_result(self, value: JsonObject | None) -> None:
        self.require_repair().semantic_suspicion_result = value

    @property
    def corrected_semantic_suspicion_result(self) -> JsonObject | None:
        return self.repair.corrected_semantic_suspicion_result if self.repair is not None else None

    @corrected_semantic_suspicion_result.setter
    def corrected_semantic_suspicion_result(self, value: JsonObject | None) -> None:
        self.require_repair().corrected_semantic_suspicion_result = value

    @property
    def correction_used(self) -> bool:
        return self.repair.correction_used if self.repair is not None else False

    @correction_used.setter
    def correction_used(self, value: bool) -> None:
        self.require_repair().correction_used = value

    @property
    def output_receipt(self) -> JsonObject:
        return _required(self.require_finalized().output_receipt, "output receipt")

    @output_receipt.setter
    def output_receipt(self, value: JsonObject) -> None:
        self.require_finalized().output_receipt = value

    @property
    def categorization_result(self) -> JsonObject | None:
        return self.finalized.categorization_result if self.finalized is not None else None

    @categorization_result.setter
    def categorization_result(self, value: JsonObject | None) -> None:
        self.require_finalized().categorization_result = value

    @property
    def categorized_receipt(self) -> JsonObject:
        return _required(self.require_finalized().categorized_receipt, "categorized receipt")

    @categorized_receipt.setter
    def categorized_receipt(self, value: JsonObject) -> None:
        self.require_finalized().categorized_receipt = value

    @property
    def pipeline_meta(self) -> JsonObject:
        return _required(self.require_finalized().pipeline_meta, "pipeline metadata")

    @pipeline_meta.setter
    def pipeline_meta(self, value: JsonObject) -> None:
        self.require_finalized().pipeline_meta = value

    @property
    def duration_seconds(self) -> float:
        return round(time.perf_counter() - self.started_at, 2)

    def as_result(self) -> dict[str, Any]:
        return {
            "receipt": self.categorized_receipt,
            "report": self.final_report,
            "paths": self.paths,
            "logs": [],
            "pipeline_meta": self.pipeline_meta,
            "observability": {
                "stage_trace": self.stage_trace,
                "metrics_path": self.paths.get("extraction_metrics"),
            },
        }
