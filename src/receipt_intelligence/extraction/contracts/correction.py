"""Typed outputs of the specialist correction coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from receipt_intelligence.extraction.contracts.common import JsonObject, StageArtifact
from receipt_intelligence.extraction.contracts.validation import ValidationReport


class CorrectionAttemptStatus(StrEnum):
    ACCEPTED = "accepted"
    ACCEPTED_REVIEW_REQUIRED = "accepted_review_required"
    ABSTAINED = "abstained"
    INVALID_JSON = "invalid_json"
    SCHEMA_INVALID = "schema_invalid"
    EVIDENCE_NOT_GROUNDED = "evidence_not_grounded"
    PROPOSAL_REJECTED = "proposal_rejected"
    TARGET_EXHAUSTED = "target_exhausted"
    OPEN_NO_STRATEGY = "open_no_strategy"
    NOT_ATTEMPTED_ROUND_LIMIT = "not_attempted_round_limit"


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
    artifacts: tuple[StageArtifact, ...] = ()

    @property
    def changed(self) -> bool:
        return self.original_receipt != self.accepted_receipt

    @property
    def corrected_codes(self) -> frozenset[str]:
        return frozenset(
            outcome.target_code
            for outcome in self.target_outcomes
            if outcome.status
            in {
                CorrectionAttemptStatus.ACCEPTED,
                CorrectionAttemptStatus.ACCEPTED_REVIEW_REQUIRED,
            }
        )


__all__ = [
    "CorrectionAttempt",
    "CorrectionAttemptStatus",
    "CorrectionResult",
    "CorrectionTargetOutcome",
]
