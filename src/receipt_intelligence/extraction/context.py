"""Mutable state passed through receipt extraction stages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.extraction.config import ExtractionConfig
from receipt_intelligence.observability.timing import utc_now_iso


@dataclass(slots=True)
class ExtractionContext:
    config: ExtractionConfig
    started_at: float = field(default_factory=time.perf_counter)
    started_at_utc: str = field(default_factory=utc_now_iso)
    paths: dict[str, Path] = field(default_factory=dict)
    stage_trace: list[dict[str, Any]] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)

    preliminary_ocr_context: dict[str, Any] | None = None
    visual_result: dict[str, Any] | None = None
    visual_evidence: dict[str, Any] | None = None
    region_reocr_result: dict[str, Any] | None = None
    table_interpretation_result: dict[str, Any] | None = None
    table_arbitration_result: dict[str, Any] | None = None

    llm_result: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    ocr_context: dict[str, Any] | None = None
    compact_evidence: dict[str, Any] | None = None
    grouped_evidence: dict[str, Any] | None = None
    table_assembly_report: dict[str, Any] = field(
        default_factory=lambda: {"attempted": False, "changed": False}
    )
    postprocess_actions: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] | None = None
    initial_report: dict[str, Any] | None = None

    reocr_result: dict[str, Any] | None = None
    right_column_recovery_result: dict[str, Any] | None = None
    vertical_price_stack_recovery_result: dict[str, Any] | None = None
    patch_correction_result: dict[str, Any] | None = None
    corrected_report: dict[str, Any] | None = None
    correction_used: bool = False

    final_receipt: dict[str, Any] | None = None
    final_report: dict[str, Any] | None = None
    output_receipt: dict[str, Any] | None = None
    categorization_result: dict[str, Any] | None = None
    categorized_receipt: dict[str, Any] | None = None
    pipeline_meta: dict[str, Any] | None = None

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

    def require(self, attribute: str) -> Any:
        value = getattr(self, attribute)
        if value is None:
            raise RuntimeError(f"Extraction stage requires context.{attribute}")
        return value

    @property
    def duration_seconds(self) -> float:
        return round(time.perf_counter() - self.started_at, 2)

    def as_result(self) -> dict[str, Any]:
        return {
            "receipt": self.require("categorized_receipt"),
            "report": self.require("final_report"),
            "paths": self.paths,
            "logs": [],
            "pipeline_meta": self.require("pipeline_meta"),
            "observability": {
                "stage_trace": self.stage_trace,
                "metrics_path": self.paths.get("extraction_metrics"),
            },
        }
