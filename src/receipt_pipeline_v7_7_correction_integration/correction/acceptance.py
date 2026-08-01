from __future__ import annotations

from typing import Any, Iterable


def failed_checks(validation: dict[str, Any]) -> list[dict[str, Any]]:
    checks = validation.get("checks")
    if not isinstance(checks, list):
        return []
    return [
        check
        for check in checks
        if isinstance(check, dict) and check.get("status") == "failed"
    ]


def failed_codes(validation: dict[str, Any]) -> set[str]:
    return {
        str(check.get("code"))
        for check in failed_checks(validation)
        if check.get("code")
    }


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


def evaluate_candidate(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    targeted_codes: Iterable[str],
) -> tuple[bool, list[str]]:
    """Require target resolution and reject all deterministic regressions."""
    reasons: list[str] = []
    before_map = _check_map(before)
    after_map = _check_map(after)
    before_failed = failed_codes(before)
    after_failed = failed_codes(after)

    for code in sorted(set(str(value) for value in targeted_codes)):
        status = (after_map.get(code) or {}).get("status")
        if status not in {"passed", "observed"}:
            reasons.append(f"target_not_resolved:{code}:status={status}")

    regressed_passed = sorted(
        code
        for code, check in before_map.items()
        if check.get("status") == "passed"
        and (after_map.get(code) or {}).get("status") != "passed"
    )
    if regressed_passed:
        reasons.append(
            "previously_passed_checks_regressed:" + ",".join(regressed_passed)
        )

    newly_failed = sorted(after_failed - before_failed)
    non_dependency_failures = [
        code
        for code in newly_failed
        if (before_map.get(code) or {}).get("status") != "skipped"
    ]
    if non_dependency_failures:
        reasons.append(
            "new_non_dependency_failures:" + ",".join(non_dependency_failures)
        )

    if len(after_failed) > len(before_failed):
        reasons.append("failed_check_count_increased")
    if _failed_severity_score(after) > _failed_severity_score(before):
        reasons.append("failed_severity_score_worsened")
    if _status_count(after, "skipped") > _status_count(before, "skipped"):
        reasons.append("skipped_check_count_increased")

    return not reasons, reasons
