"""Typed contracts for validator-gated specialist correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from receipt_intelligence.application.ports.llm import ModelCallMetrics
from receipt_intelligence.extraction.contracts.common import JsonObject, StageArtifact
from receipt_intelligence.extraction.contracts.transcription import TranscriptionResult
from receipt_intelligence.extraction.contracts.validation import ValidationReport


class CorrectionAttemptStatus(StrEnum):
    ACCEPTED = "accepted"
    ABSTAINED = "abstained"
    INVALID_JSON = "invalid_json"
    INVALID_EVIDENCE = "invalid_evidence"
    INVALID_PATCH = "invalid_patch"
    REJECTED_NO_IMPROVEMENT = "rejected_no_improvement"
    ERROR = "error"
    TARGET_EXHAUSTED = "target_exhausted"
    OPEN_NO_STRATEGY = "open_no_strategy"


@dataclass(frozen=True, slots=True)
class CorrectionRequest:
    run_id: str
    receipt: JsonObject
    transcription: TranscriptionResult
    validation: ValidationReport
    item_contract: JsonObject
    item_pipeline_enabled: bool
    selected_scalar_tasks: tuple[str, ...]

    def __post_init__(self) -> None:
        run_id = str(self.run_id or "").strip()
        if not run_id:
            raise ValueError("CorrectionRequest.run_id must not be empty.")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "receipt", dict(self.receipt))
        object.__setattr__(self, "item_contract", dict(self.item_contract))
        object.__setattr__(
            self,
            "selected_scalar_tasks",
            tuple(str(value).strip() for value in self.selected_scalar_tasks if str(value).strip()),
        )


@dataclass(frozen=True, slots=True)
class CorrectionAttempt:
    target_code: str
    strategy_id: str
    status: CorrectionAttemptStatus
    receipt_modified: bool = False
    diagnostics: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CorrectionTargetOutcome:
    target_code: str
    status: CorrectionAttemptStatus
    related_codes: frozenset[str] = frozenset()
    attempt_count: int = 0


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    original_receipt: JsonObject
    accepted_receipt: JsonObject
    final_validation: ValidationReport
    attempts: tuple[CorrectionAttempt, ...] = ()
    target_outcomes: tuple[CorrectionTargetOutcome, ...] = ()
    report: JsonObject = field(default_factory=dict)
    model_calls: tuple[ModelCallMetrics, ...] = ()
    artifacts: tuple[StageArtifact, ...] = ()

    @property
    def changed(self) -> bool:
        return self.original_receipt != self.accepted_receipt

    @property
    def corrected_codes(self) -> frozenset[str]:
        return frozenset(
            outcome.target_code
            for outcome in self.target_outcomes
            if outcome.status is CorrectionAttemptStatus.ACCEPTED
        )


__all__ = [
    "CorrectionAttempt",
    "CorrectionAttemptStatus",
    "CorrectionRequest",
    "CorrectionResult",
    "CorrectionTargetOutcome",
]
