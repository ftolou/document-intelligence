from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .acceptance import evaluate_candidate, failed_checks, failed_codes, validation_score
from .normalization import normalize_source_evidence
from .patching import apply_patch, target_for_strategy, validate_patch
from .profile import CorrectionProfile, StrategyConfig
from .strategies import (
    build_final_total_patch,
    build_item_sum_patch,
    build_vat_patch,
    validate_final_total_evidence,
    validate_item_sum_evidence,
    validate_vat_evidence,
)


@dataclass(frozen=True)
class CorrectionCallbacks:
    invoke_source_evidence: Callable[[StrategyConfig, str, int, int], dict[str, Any]]
    validate_receipt: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]]
    effective_item_pipeline: Callable[
        [dict[str, Any] | None, dict[str, Any]], dict[str, Any] | None
    ]
    write_artifact: Callable[[str, Any], None]


_MISSING_FINAL_PRICE_CODE = "MISSING_FINAL_PRICE"
_MISSING_PRICE_TARGET_CODES = frozenset({"ITEM_CONTRACT", "ITEM_PRICES_COMPLETE"})


def _item_contract_is_missing_final_price_only(check: dict[str, Any]) -> bool:
    if str(check.get("code") or "") != "ITEM_CONTRACT":
        return False
    details = check.get("details")
    return (
        bool(details)
        and isinstance(details, list)
        and all(
            isinstance(detail, dict) and str(detail.get("code") or "") == _MISSING_FINAL_PRICE_CODE
            for detail in details
        )
    )


def _strategy_routing_reason(
    strategy_id: str,
    check: dict[str, Any],
) -> str | None:
    if strategy_id != "item_sum_source_blocks_v3":
        return None
    code = str(check.get("code") or "")
    if code == "ITEM_CONTRACT" and not _item_contract_is_missing_final_price_only(check):
        return (
            "The V3 item source-evidence strategy is eligible for ITEM_CONTRACT "
            "only when every contract detail is MISSING_FINAL_PRICE."
        )
    return None


def _related_target_codes(
    check: dict[str, Any],
    validation: dict[str, Any],
) -> set[str]:
    code = str(check.get("code") or "UNNAMED_CONSTRAINT")
    related = {code}
    if code not in _MISSING_PRICE_TARGET_CODES:
        return related

    failed_by_code = {
        str(candidate.get("code") or "UNNAMED_CONSTRAINT"): candidate
        for candidate in failed_checks(validation)
    }
    contract = failed_by_code.get("ITEM_CONTRACT")
    prices = failed_by_code.get("ITEM_PRICES_COMPLETE")
    if (
        contract is not None
        and prices is not None
        and _item_contract_is_missing_final_price_only(contract)
    ):
        related.update(_MISSING_PRICE_TARGET_CODES)
    return related


def _select_check(
    validation: dict[str, Any],
    excluded_codes: set[str],
) -> dict[str, Any] | None:
    candidates = [
        check
        for check in failed_checks(validation)
        if str(check.get("code") or "UNNAMED_CONSTRAINT") not in excluded_codes
    ]
    if not candidates:
        return None
    severity_rank = {"error": 0, "review": 1, "warning": 2, "info": 3}
    return min(
        enumerate(candidates),
        key=lambda pair: (
            severity_rank.get(str(pair[1].get("severity")), 9),
            pair[0],
        ),
    )[1]


def _evidence_to_patch(
    strategy_id: str,
    answer: Any,
    transcription: str,
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if strategy_id == "item_sum_source_blocks_v3":
        evidence_validation = validate_item_sum_evidence(answer, transcription)
        if evidence_validation.get("status") != "valid":
            return {"patches": []}, evidence_validation, {"status": "not_built"}
        patch, diagnostics = build_item_sum_patch(answer, receipt)
        return patch, evidence_validation, diagnostics
    if strategy_id == "vat_source_evidence_v9":
        evidence_validation = validate_vat_evidence(answer, transcription)
        if evidence_validation.get("status") != "valid":
            return {"patches": []}, evidence_validation, {"status": "not_built"}
        patch, diagnostics = build_vat_patch(answer, receipt)
        return patch, evidence_validation, diagnostics
    if strategy_id == "final_total_source_evidence_v2_4":
        evidence_validation = validate_final_total_evidence(answer, transcription)
        if evidence_validation.get("status") != "valid":
            return {"patches": []}, evidence_validation, {"status": "not_built"}
        patch, diagnostics = build_final_total_patch(answer, receipt)
        return patch, evidence_validation, diagnostics
    raise ValueError(f"Unsupported source-evidence strategy: {strategy_id}")


def run_correction_coordinator(
    *,
    profile: CorrectionProfile,
    callbacks: CorrectionCallbacks,
    transcription: str,
    receipt: dict[str, Any],
    initial_validation: dict[str, Any],
    item_pipeline_result: dict[str, Any] | None,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    report: dict[str, Any] = {
        "mode": "validator_gated_specialist_coordinator",
        "profile": profile.as_dict(),
        "triggered": False,
        "applied": False,
        "receipt_unchanged": True,
        "initial_validation_status": initial_validation.get("status"),
        "initial_validation_score": list(validation_score(initial_validation)),
        "rounds": [],
        "attempts": [],
        "target_outcomes": [],
        "accepted_patches": [],
        "accepted_corrections": [],
    }

    current_receipt = copy.deepcopy(receipt)
    current_validation = copy.deepcopy(initial_validation)
    current_item_pipeline = copy.deepcopy(item_pipeline_result)
    exhausted_target_codes: set[str] = set()
    open_no_strategy_codes: set[str] = set()
    corrected_target_codes: set[str] = set()

    def finish(status: str, **extra: Any):
        remaining = sorted(failed_codes(current_validation))
        attempt_status_counts = Counter(
            str(attempt.get("status") or "unknown") for attempt in report["attempts"]
        )
        normalization_records = [
            attempt.get("normalization")
            for attempt in report["attempts"]
            if isinstance(attempt.get("normalization"), dict)
        ]
        normalized_records = [
            value for value in normalization_records if value.get("status") == "normalized"
        ]
        json_repair_records = [
            attempt.get("json_repair")
            for attempt in report["attempts"]
            if isinstance(attempt.get("json_repair"), dict)
        ]
        triggered_json_repairs = [value for value in json_repair_records if value.get("triggered")]
        completed_json_repairs = [
            value for value in triggered_json_repairs if value.get("status") == "completed"
        ]
        report.update(
            {
                "status": status,
                "applied": bool(report["accepted_patches"]),
                "receipt_unchanged": not bool(report["accepted_patches"]),
                "final_validation_status": current_validation.get("status"),
                "final_validation_score": list(validation_score(current_validation)),
                "remaining_failed_codes": remaining,
                "corrected_target_codes": sorted(corrected_target_codes),
                "exhausted_target_codes": sorted(exhausted_target_codes),
                "open_no_strategy_codes": sorted(open_no_strategy_codes),
                "unattempted_failed_codes": sorted(
                    set(remaining) - exhausted_target_codes - open_no_strategy_codes
                ),
                "attempt_status_counts": dict(sorted(attempt_status_counts.items())),
                "normalization_summary": {
                    "attempt_count": len(normalization_records),
                    "normalized_attempt_count": len(normalized_records),
                    "operation_count": sum(
                        int(value.get("operation_count") or 0) for value in normalized_records
                    ),
                },
                "json_repair_summary": {
                    "record_count": len(json_repair_records),
                    "triggered_count": len(triggered_json_repairs),
                    "completed_count": len(completed_json_repairs),
                    "failed_count": sum(
                        1 for value in triggered_json_repairs if value.get("status") != "completed"
                    ),
                },
                "summary": {
                    "accepted_correction_count": len(report["accepted_corrections"]),
                    "accepted_patch_count": len(report["accepted_patches"]),
                    "target_round_count": len(report["rounds"]),
                    "exhausted_target_count": len(exhausted_target_codes),
                    "open_no_strategy_count": len(open_no_strategy_codes),
                    "remaining_failed_count": len(remaining),
                },
                **extra,
            }
        )
        return current_receipt, current_validation, current_item_pipeline, report

    if not enabled or not profile.automatic_patching:
        return finish("disabled")
    if not failed_checks(current_validation):
        return finish("skipped_validation_clean")

    report["triggered"] = True
    for round_index in range(1, profile.max_rounds + 1):
        check = _select_check(current_validation, exhausted_target_codes)
        if check is None:
            if failed_checks(current_validation):
                return finish(
                    "accepted_partial_open_failures"
                    if report["accepted_patches"]
                    else "open_failures_unresolved"
                )
            return finish("accepted_all_targets_resolved")

        code = str(check.get("code") or "UNNAMED_CONSTRAINT")
        configured_chain = profile.strategy_chain(code)
        ineligible_strategies = [
            {
                "strategy_id": strategy.strategy_id,
                "reason": reason,
            }
            for strategy in configured_chain
            if (reason := _strategy_routing_reason(strategy.strategy_id, check)) is not None
        ]
        ineligible_ids = {entry["strategy_id"] for entry in ineligible_strategies}
        chain = tuple(
            strategy for strategy in configured_chain if strategy.strategy_id not in ineligible_ids
        )
        targeted_codes = _related_target_codes(check, current_validation)
        round_record: dict[str, Any] = {
            "round": round_index,
            "target_code": code,
            "target_codes": sorted(targeted_codes),
            "target_check": copy.deepcopy(check),
            "configured_strategy_chain": [strategy.strategy_id for strategy in configured_chain],
            "strategy_chain": [strategy.strategy_id for strategy in chain],
            "ineligible_strategies": ineligible_strategies,
            "validation_status_before": current_validation.get("status"),
            "validation_score_before": list(validation_score(current_validation)),
            "strategies": [],
        }
        report["rounds"].append(round_record)

        if not chain:
            exhausted_target_codes.add(code)
            open_no_strategy_codes.add(code)
            round_record.update(
                {
                    "status": "open_no_strategy",
                    "receipt_modified": False,
                }
            )
            report["target_outcomes"].append(
                {
                    "round": round_index,
                    "target_code": code,
                    "target_codes": [code],
                    "status": "open_no_strategy",
                    "ineligible_strategies": copy.deepcopy(ineligible_strategies),
                    "receipt_modified": False,
                }
            )
            continue

        accepted = False
        for strategy_index, strategy in enumerate(chain, start=1):
            target = target_for_strategy(
                strategy.strategy_id,
                check,
                current_validation,
                current_receipt,
                max_patches=strategy.max_patches,
            )
            target["targeted_codes"] = sorted(targeted_codes)
            strategy_record: dict[str, Any] = {
                "strategy_index": strategy_index,
                "strategy_id": strategy.strategy_id,
                "kind": strategy.kind,
                "target": copy.deepcopy(target),
                "attempts": [],
            }
            round_record["strategies"].append(strategy_record)
            if strategy.kind != "source_evidence":
                strategy_record.update(
                    {
                        "status": "unsupported_strategy_kind",
                        "reason": "Only source-evidence correction strategies are enabled.",
                    }
                )
                continue
            if not target.get("model_patch_supported"):
                strategy_record.update(
                    {
                        "status": "unsupported_scope",
                        "reason": target.get("routing_reason"),
                    }
                )
                continue

            previous_errors: list[dict[str, Any]] = []
            for attempt in range(1, strategy.max_attempts + 1):
                prefix = (
                    f"90_correction_round_{round_index:02d}_"
                    f"strategy_{strategy_index:02d}_attempt_{attempt:02d}"
                )
                attempt_record: dict[str, Any] = {
                    "round": round_index,
                    "strategy_id": strategy.strategy_id,
                    "attempt": attempt,
                    "target_code": code,
                    "target_codes": sorted(targeted_codes),
                    "receipt_modified": False,
                }
                strategy_record["attempts"].append(attempt_record)
                report["attempts"].append(attempt_record)
                try:
                    result = callbacks.invoke_source_evidence(
                        strategy, transcription, round_index, attempt
                    )
                    raw_answer = result.get("answer")
                    callbacks.write_artifact(f"{prefix}_source_evidence_result.json", result)
                    attempt_record.update(
                        {
                            "model_metrics": result.get("metrics"),
                            "json_repair": copy.deepcopy(result.get("json_repair")),
                        }
                    )
                    result_status = result.get("status")
                    if result_status is None and "answer" in result:
                        result_status = "completed"
                    if result_status != "completed":
                        error = result.get("error") or {
                            "code": "SOURCE_EVIDENCE_OUTPUT_INVALID",
                            "message": f"Source-evidence call status: {result_status}",
                        }
                        previous_errors = [copy.deepcopy(error)]
                        attempt_record.update(
                            {
                                "status": str(result_status or "error"),
                                "error": copy.deepcopy(error),
                            }
                        )
                        if attempt < strategy.max_attempts:
                            continue
                        strategy_record.update(
                            {
                                "status": str(result_status or "error"),
                                "errors": previous_errors,
                            }
                        )
                        break
                    normalized_answer, normalization = normalize_source_evidence(
                        strategy.strategy_id,
                        raw_answer,
                        transcription,
                    )
                    callbacks.write_artifact(f"{prefix}_evidence_normalization.json", normalization)
                    if normalization.get("status") == "normalized":
                        callbacks.write_artifact(
                            f"{prefix}_source_evidence_normalized.json",
                            normalized_answer,
                        )
                    attempt_record.update(
                        {
                            "normalization": normalization,
                        }
                    )

                    patch_answer, evidence_validation, adapter_diagnostics = _evidence_to_patch(
                        strategy.strategy_id,
                        normalized_answer,
                        transcription,
                        current_receipt,
                    )
                    callbacks.write_artifact(
                        f"{prefix}_evidence_validation.json", evidence_validation
                    )
                    callbacks.write_artifact(
                        f"{prefix}_adapter_diagnostics.json", adapter_diagnostics
                    )
                    attempt_record.update(
                        {
                            "evidence_validation": evidence_validation,
                            "adapter_diagnostics": adapter_diagnostics,
                        }
                    )
                    if evidence_validation.get("status") != "valid":
                        previous_errors = evidence_validation.get("errors") or []
                        attempt_record["status"] = "invalid_evidence"
                        if attempt < strategy.max_attempts:
                            continue
                        strategy_record.update(
                            {"status": "invalid_evidence", "errors": previous_errors}
                        )
                        break

                    answer = patch_answer
                    callbacks.write_artifact(f"{prefix}_patch.json", answer)
                    patch_validation = validate_patch(
                        answer,
                        current_receipt,
                        target=target,
                    )
                    callbacks.write_artifact(f"{prefix}_patch_validation.json", patch_validation)
                    attempt_record["patch_validation"] = patch_validation
                    attempt_record["patch"] = copy.deepcopy(answer)
                    if patch_validation.get("status") != "valid":
                        previous_errors = patch_validation.get("errors") or []
                        attempt_record["status"] = "invalid_patch"
                        if attempt < strategy.max_attempts:
                            continue
                        strategy_record.update(
                            {"status": "invalid_patch", "errors": previous_errors}
                        )
                        break

                    patches = answer.get("patches") if isinstance(answer, dict) else None
                    if not patches:
                        attempt_record["status"] = "abstained"
                        strategy_record["status"] = "abstained"
                        break

                    candidate_receipt = apply_patch(current_receipt, answer)
                    candidate_item_pipeline = callbacks.effective_item_pipeline(
                        current_item_pipeline,
                        candidate_receipt,
                    )
                    candidate_validation = callbacks.validate_receipt(
                        candidate_receipt,
                        candidate_item_pipeline,
                    )
                    callbacks.write_artifact(f"{prefix}_candidate_receipt.json", candidate_receipt)
                    callbacks.write_artifact(
                        f"{prefix}_candidate_validation.json", candidate_validation
                    )
                    improves, rejection_reasons = evaluate_candidate(
                        current_validation,
                        candidate_validation,
                        targeted_codes=targeted_codes,
                    )
                    attempt_record.update(
                        {
                            "candidate_validation_status": candidate_validation.get("status"),
                            "candidate_validation_score": list(
                                validation_score(candidate_validation)
                            ),
                            "accepted": improves,
                            "rejection_reasons": rejection_reasons,
                        }
                    )
                    if improves:
                        candidate_validation = copy.deepcopy(candidate_validation)
                        policy = candidate_validation.get("policy")
                        if isinstance(policy, dict):
                            policy["changes_model_values"] = True
                            policy["correction_applied"] = True
                        current_receipt = candidate_receipt
                        current_validation = candidate_validation
                        current_item_pipeline = candidate_item_pipeline
                        report["accepted_patches"].extend(copy.deepcopy(patches))
                        report["accepted_corrections"].append(
                            {
                                "round": round_index,
                                "strategy_id": strategy.strategy_id,
                                "target_code": code,
                                "target_codes": sorted(targeted_codes),
                                "patches": copy.deepcopy(patches),
                                "prompt": result.get("request", {}).get("prompt"),
                                "metrics": result.get("metrics"),
                                "normalization": copy.deepcopy(normalization),
                            }
                        )
                        corrected_target_codes.update(targeted_codes)
                        attempt_record.update(
                            {
                                "status": "accepted",
                                "receipt_modified": True,
                            }
                        )
                        strategy_record["status"] = "accepted"
                        round_record.update(
                            {
                                "status": "accepted",
                                "accepted_strategy": strategy.strategy_id,
                                "accepted_attempt": attempt,
                                "validation_status_after": current_validation.get("status"),
                                "validation_score_after": list(
                                    validation_score(current_validation)
                                ),
                                "receipt_modified": True,
                            }
                        )
                        report["target_outcomes"].append(
                            {
                                "round": round_index,
                                "target_code": code,
                                "target_codes": sorted(targeted_codes),
                                "status": "accepted",
                                "strategy_id": strategy.strategy_id,
                                "attempt": attempt,
                                "receipt_modified": True,
                            }
                        )
                        accepted = True
                        break

                    previous_errors = [
                        {
                            "code": "PATCH_DID_NOT_RESOLVE_TARGET",
                            "target_code": code,
                            "target_codes": sorted(targeted_codes),
                            "reasons": rejection_reasons,
                        }
                    ]
                    attempt_record["status"] = "rejected_no_improvement"
                    if attempt < strategy.max_attempts:
                        continue
                    strategy_record.update(
                        {
                            "status": "rejected_no_improvement",
                            "rejection_reasons": rejection_reasons,
                        }
                    )
                    break
                except Exception as exc:
                    error = {
                        "code": "CORRECTION_CALL_FAILED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    attempt_record.update({"status": "error", **error})
                    previous_errors = [error]
                    if attempt < strategy.max_attempts:
                        continue
                    strategy_record.update({"status": "error", "errors": [error]})
                    break

            if accepted:
                break

        if accepted:
            continue

        exhausted_target_codes.update(targeted_codes)
        round_record.update(
            {
                "status": "target_exhausted",
                "receipt_modified": False,
            }
        )
        report["target_outcomes"].append(
            {
                "round": round_index,
                "target_code": code,
                "target_codes": sorted(targeted_codes),
                "status": "target_exhausted",
                "strategy_ids": [strategy.strategy_id for strategy in chain],
                "receipt_modified": False,
            }
        )

    if failed_codes(current_validation):
        return finish(
            "accepted_partial_max_rounds"
            if report["accepted_patches"]
            else "max_rounds_open_failures"
        )
    return finish("accepted_all_targets_resolved")
