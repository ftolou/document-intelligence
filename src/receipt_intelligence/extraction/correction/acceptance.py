from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def failed_checks(validation: dict[str, Any]) -> list[dict[str, Any]]:
    checks = validation.get("checks")
    if not isinstance(checks, list):
        return []
    return [
        check for check in checks if isinstance(check, dict) and check.get("status") == "failed"
    ]


def failed_codes(validation: dict[str, Any]) -> set[str]:
    return {str(check.get("code")) for check in failed_checks(validation) if check.get("code")}


def validation_score(validation: dict[str, Any]) -> tuple[int, int, int]:
    summary = validation.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    return (
        int(summary.get("error_count") or 0),
        int(summary.get("review_count") or 0),
        int(summary.get("failed") or 0),
    )


def _check_map(validation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = validation.get("checks")
    checks = checks if isinstance(checks, list) else []
    return {
        str(check.get("code")): check
        for check in checks
        if isinstance(check, dict) and check.get("code")
    }


def _failed_severity_score(validation: dict[str, Any]) -> tuple[int, int, int]:
    failed = failed_checks(validation)
    return (
        sum(1 for check in failed if check.get("severity") == "error"),
        sum(1 for check in failed if check.get("severity") == "review"),
        len(failed),
    )


def _status_count(validation: dict[str, Any], name: str) -> int:
    summary = validation.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    return int(summary.get(name) or 0)


_RESIDUAL_FIELDS = {
    "ITEM_SUM_RECONCILIATION": ("absolute_direct_difference",),
    "PRE_DISCOUNT_TOTAL_RECONCILIATION": ("difference",),
    "NET_PLUS_VAT_RECONCILIATION": ("difference",),
    "VAT_LINE_SUM_RECONCILIATION": ("difference",),
    "VAT_LINES_GROSS_RECONCILIATION": ("difference",),
    "PAYMENT_CHANGE_RECONCILIATION": ("difference",),
}


def _numeric_residual(check: dict[str, Any]) -> float | None:
    values = check.get("values")
    values = values if isinstance(values, dict) else {}
    for field in _RESIDUAL_FIELDS.get(str(check.get("code")), ()):
        value = values.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return abs(float(value))
    return None


def _failed_target_improved(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if before.get("status") != "failed" or after.get("status") != "failed":
        return False
    before_residual = _numeric_residual(before)
    after_residual = _numeric_residual(after)
    return (
        before_residual is not None
        and after_residual is not None
        and after_residual < before_residual
    )


def evaluate_candidate(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    targeted_codes: Iterable[str],
    allow_partial_improvement: bool = False,
) -> tuple[bool, list[str]]:
    """Require target progress and reject all deterministic regressions."""
    reasons: list[str] = []
    before_map = _check_map(before)
    after_map = _check_map(after)
    before_failed = failed_codes(before)
    after_failed = failed_codes(after)

    for code in sorted(set(str(value) for value in targeted_codes)):
        after_check = after_map.get(code) or {}
        status = after_check.get("status")
        partially_improved = allow_partial_improvement and _failed_target_improved(
            before_map.get(code) or {}, after_check
        )
        if status not in {"passed", "observed"} and not partially_improved:
            reasons.append(f"target_not_resolved:{code}:status={status}")

    regressed_passed = sorted(
        code
        for code, check in before_map.items()
        if check.get("status") == "passed" and (after_map.get(code) or {}).get("status") != "passed"
    )
    if regressed_passed:
        reasons.append("previously_passed_checks_regressed:" + ",".join(regressed_passed))

    newly_failed = sorted(after_failed - before_failed)
    non_dependency_failures = [
        code for code in newly_failed if (before_map.get(code) or {}).get("status") != "skipped"
    ]
    if non_dependency_failures:
        reasons.append("new_non_dependency_failures:" + ",".join(non_dependency_failures))

    if len(after_failed) > len(before_failed):
        reasons.append("failed_check_count_increased")
    if _failed_severity_score(after) > _failed_severity_score(before):
        reasons.append("failed_severity_score_worsened")
    if _status_count(after, "skipped") > _status_count(before, "skipped"):
        reasons.append("skipped_check_count_increased")

    return not reasons, reasons
