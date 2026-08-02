"""Typed application service for the v7.8 specialist correction coordinator."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any

from receipt_intelligence.extraction.contracts.correction import (
    CorrectionAttempt,
    CorrectionAttemptStatus,
    CorrectionRequest,
    CorrectionResult,
    CorrectionTargetOutcome,
)
from receipt_intelligence.extraction.contracts.validation import ValidationReport, ValidationRequest
from receipt_intelligence.extraction.correction.artifacts import (
    CorrectionArtifactSink,
    NullCorrectionArtifactSink,
)
from receipt_intelligence.extraction.correction.core import CorrectionCallbacks, run_correction_coordinator
from receipt_intelligence.extraction.correction.invocation import SourceEvidenceInvoker
from receipt_intelligence.extraction.correction.profile import CorrectionProfile
from receipt_intelligence.extraction.services.correction import ReceiptCorrectionService
from receipt_intelligence.extraction.services.validation import ReceiptValidationService
from receipt_intelligence.extraction.structured.item_contract import validate_direct_items


_STATUS_MAP = {
    "accepted": CorrectionAttemptStatus.ACCEPTED,
    "abstained": CorrectionAttemptStatus.ABSTAINED,
    "invalid_json": CorrectionAttemptStatus.INVALID_JSON,
    "schema_invalid": CorrectionAttemptStatus.INVALID_JSON,
    "invalid_evidence": CorrectionAttemptStatus.INVALID_EVIDENCE,
    "invalid_patch": CorrectionAttemptStatus.INVALID_PATCH,
    "rejected_no_improvement": CorrectionAttemptStatus.REJECTED_NO_IMPROVEMENT,
    "error": CorrectionAttemptStatus.ERROR,
    "target_exhausted": CorrectionAttemptStatus.TARGET_EXHAUSTED,
    "open_no_strategy": CorrectionAttemptStatus.OPEN_NO_STRATEGY,
}


class SpecialistCorrectionService(ReceiptCorrectionService):
    def __init__(
        self,
        *,
        profile: CorrectionProfile,
        invoker: SourceEvidenceInvoker,
        validation_service: ReceiptValidationService,
        artifact_sink: CorrectionArtifactSink | None = None,
        enabled: bool = True,
    ) -> None:
        self._profile = profile
        self._invoker = invoker
        self._validation = validation_service
        self._sink = artifact_sink or NullCorrectionArtifactSink()
        self._enabled = enabled

    def correct(self, request: CorrectionRequest) -> CorrectionResult:
        initial_item_pipeline = {
            "status": request.item_contract.get("status"),
            "items": copy.deepcopy(request.receipt.get("items") or []),
            "validation": copy.deepcopy(request.item_contract),
        }

        def validate_candidate(
            candidate_receipt: dict[str, Any],
            candidate_item_pipeline: dict[str, Any] | None,
        ) -> dict[str, Any]:
            item_contract = (
                candidate_item_pipeline.get("validation")
                if isinstance(candidate_item_pipeline, dict)
                and isinstance(candidate_item_pipeline.get("validation"), dict)
                else request.item_contract
            )
            report = self._validation.validate(
                ValidationRequest(
                    receipt=candidate_receipt,
                    item_contract=item_contract,
                    item_pipeline_enabled=request.item_pipeline_enabled,
                    selected_scalar_tasks=request.selected_scalar_tasks,
                    money_tolerance=float(
                        request.validation.raw.get("policy", {}).get("money_tolerance", 0.02)
                    ),
                    vat_rate_tolerance=float(
                        request.validation.raw.get("policy", {}).get("vat_rate_tolerance", 0.02)
                    ),
                )
            )
            return report.to_dict()

        def effective_item_pipeline(
            previous: dict[str, Any] | None,
            candidate_receipt: dict[str, Any],
        ) -> dict[str, Any] | None:
            if not request.item_pipeline_enabled:
                return previous
            items = candidate_receipt.get("items")
            answer = {"items": items if isinstance(items, list) else []}
            contract = validate_direct_items(answer)
            result = copy.deepcopy(previous) if isinstance(previous, dict) else {}
            result.update(
                {
                    "status": (
                        "completed"
                        if contract.get("status") != "invalid"
                        else "completed_with_errors"
                    ),
                    "strategy": "complete_receipt_direct_item_extraction_with_correction_overlay",
                    "items": answer["items"],
                    "validation": contract,
                    "correction_overlay": True,
                }
            )
            return result

        callbacks = CorrectionCallbacks(
            invoke_source_evidence=self._invoker.invoke,
            validate_receipt=validate_candidate,
            effective_item_pipeline=effective_item_pipeline,
            write_artifact=self._sink.write_json,
        )
        accepted_receipt, final_validation_raw, _, report = run_correction_coordinator(
            profile=self._profile,
            callbacks=callbacks,
            transcription=request.transcription.canonical_text,
            receipt=request.receipt,
            initial_validation=request.validation.to_dict(),
            item_pipeline_result=initial_item_pipeline,
            enabled=self._enabled,
        )
        self._sink.write_json("90_gemma_correction_report.json", report)
        self._sink.write_json("91_deterministic_validation_final.json", final_validation_raw)
        attempts = tuple(_attempt(value) for value in report.get("attempts") or [])
        attempt_counts = Counter(attempt.target_code for attempt in attempts)
        outcomes = tuple(
            _outcome(value, attempt_counts)
            for value in report.get("target_outcomes") or []
            if isinstance(value, dict)
        )
        return CorrectionResult(
            original_receipt=request.receipt,
            accepted_receipt=accepted_receipt,
            final_validation=ValidationReport.from_legacy(final_validation_raw),
            attempts=attempts,
            target_outcomes=outcomes,
            report=report,
            artifacts=self._sink.artifacts,
        )


def _status(value: Any) -> CorrectionAttemptStatus:
    return _STATUS_MAP.get(str(value or ""), CorrectionAttemptStatus.ERROR)


def _attempt(value: Any) -> CorrectionAttempt:
    raw = value if isinstance(value, dict) else {}
    return CorrectionAttempt(
        target_code=str(raw.get("target_code") or "UNNAMED_CONSTRAINT"),
        strategy_id=str(raw.get("strategy_id") or "unassigned"),
        status=_status(raw.get("status")),
        receipt_modified=bool(raw.get("receipt_modified")),
        diagnostics=copy.deepcopy(raw),
    )


def _outcome(value: dict[str, Any], counts: Counter[str]) -> CorrectionTargetOutcome:
    target = str(value.get("target_code") or "UNNAMED_CONSTRAINT")
    return CorrectionTargetOutcome(
        target_code=target,
        status=_status(value.get("status")),
        related_codes=frozenset(str(code) for code in (value.get("target_codes") or [target])),
        attempt_count=counts[target],
    )


__all__ = ["SpecialistCorrectionService"]
