"""Typed contracts for post-validation categorization and final publication."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from receipt_intelligence.extraction.contracts.common import JsonObject, StageArtifact
from receipt_intelligence.extraction.contracts.validation import ValidationReport


class CategorizationStatus(StrEnum):
    DISABLED = "disabled"
    OK = "ok"
    OK_WITH_WARNINGS = "ok_with_warnings"
    SKIPPED_NO_ITEMS = "skipped_no_items"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CategorizationRequest:
    run_id: str
    receipt: JsonObject
    enabled: bool = True

    def __post_init__(self) -> None:
        run_id = str(self.run_id or "").strip()
        if not run_id:
            raise ValueError("CategorizationRequest.run_id must not be empty.")
        if not isinstance(self.receipt, dict):
            raise TypeError("CategorizationRequest.receipt must be an object.")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "receipt", dict(self.receipt))


@dataclass(frozen=True, slots=True)
class CategorizationResult:
    status: CategorizationStatus
    receipt: JsonObject
    categories: tuple[JsonObject, ...] = ()
    merchant_classification: JsonObject = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    prompt: str = ""
    raw_output: str = ""
    duration_seconds: float | None = None
    error: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, dict):
            raise TypeError("CategorizationResult.receipt must be an object.")
        object.__setattr__(self, "receipt", dict(self.receipt))
        object.__setattr__(
            self,
            "categories",
            tuple(dict(value) for value in self.categories if isinstance(value, dict)),
        )
        object.__setattr__(self, "merchant_classification", dict(self.merchant_classification))
        object.__setattr__(
            self,
            "warnings",
            tuple(str(value) for value in self.warnings if str(value).strip()),
        )

    def to_dict(self, *, include_model_io: bool = True) -> JsonObject:
        payload: JsonObject = {
            "status": self.status.value,
            "receipt": dict(self.receipt),
            "categories": [dict(value) for value in self.categories],
            "merchant_classification": dict(self.merchant_classification),
            "warnings": list(self.warnings),
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "model": self.model,
        }
        if include_model_io:
            payload["prompt"] = self.prompt
            payload["raw_output"] = self.raw_output
        return payload


@dataclass(frozen=True, slots=True)
class FinalizationRequest:
    run_id: str
    receipt: JsonObject
    validation: ValidationReport
    categorization: CategorizationResult
    stage_trace: tuple[JsonObject, ...] = ()
    upstream_metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        run_id = str(self.run_id or "").strip()
        if not run_id:
            raise ValueError("FinalizationRequest.run_id must not be empty.")
        if not isinstance(self.receipt, dict):
            raise TypeError("FinalizationRequest.receipt must be an object.")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "receipt", dict(self.receipt))
        object.__setattr__(
            self,
            "stage_trace",
            tuple(dict(value) for value in self.stage_trace if isinstance(value, dict)),
        )
        object.__setattr__(self, "upstream_metadata", dict(self.upstream_metadata))


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    receipt: JsonObject
    validation: ValidationReport
    categorization: CategorizationResult
    pipeline_metadata: JsonObject
    paths: dict[str, Path] = field(default_factory=dict)
    artifacts: tuple[StageArtifact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt", dict(self.receipt))
        object.__setattr__(self, "pipeline_metadata", dict(self.pipeline_metadata))
        object.__setattr__(
            self,
            "paths",
            {str(key): Path(value) for key, value in self.paths.items()},
        )

    def as_application_result(self) -> JsonObject:
        return {
            "receipt": dict(self.receipt),
            "report": self.validation.to_dict(),
            "paths": dict(self.paths),
            "logs": [],
            "pipeline_meta": dict(self.pipeline_metadata),
            "observability": {
                "stage_trace": self.pipeline_metadata.get("workflow", {}).get("stage_trace", []),
                "metrics_path": self.paths.get("extraction_metrics"),
            },
        }


__all__ = [
    "CategorizationRequest",
    "CategorizationResult",
    "CategorizationStatus",
    "FinalizationRequest",
    "FinalizationResult",
]
