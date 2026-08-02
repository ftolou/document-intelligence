"""Typed deterministic validation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from receipt_intelligence.extraction.contracts.common import JsonObject


class ValidationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    OBSERVED = "observed"


class ValidationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    code: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    details: tuple[JsonObject, ...] = ()

    def __post_init__(self) -> None:
        code = str(self.code or "").strip()
        message = str(self.message or "").strip()
        if not code:
            raise ValueError("ValidationCheck.code must not be empty.")
        if not message:
            raise ValueError("ValidationCheck.message must not be empty.")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    status: str
    checks: tuple[ValidationCheck, ...]
    raw: JsonObject = field(default_factory=dict)

    @property
    def failed_codes(self) -> frozenset[str]:
        return frozenset(
            check.code for check in self.checks if check.status is ValidationStatus.FAILED
        )

    def find(self, code: str) -> ValidationCheck | None:
        return next((check for check in self.checks if check.code == code), None)

    @classmethod
    def from_legacy(cls, report: dict[str, Any]) -> ValidationReport:
        raw_checks = report.get("checks") or report.get("validation_checks") or []
        if not raw_checks:
            raw_checks = [
                {**issue, "status": issue.get("status") or "failed"}
                for issue in (report.get("issues") or [])
                if isinstance(issue, dict)
            ]
        if isinstance(raw_checks, dict):
            raw_checks = [raw_checks]
        checks: list[ValidationCheck] = []
        for raw_check in raw_checks:
            if not isinstance(raw_check, dict):
                continue
            status_text = str(raw_check.get("status") or "observed").lower()
            severity_text = str(raw_check.get("severity") or "warning").lower()
            try:
                status = ValidationStatus(status_text)
            except ValueError:
                status = ValidationStatus.OBSERVED
            try:
                severity = ValidationSeverity(severity_text)
            except ValueError:
                severity = ValidationSeverity.WARNING
            raw_details = raw_check.get("details")
            if isinstance(raw_details, list):
                details = tuple(item for item in raw_details if isinstance(item, dict))
            elif isinstance(raw_details, dict):
                details = (raw_details,)
            else:
                details = ()
            checks.append(
                ValidationCheck(
                    code=str(raw_check.get("code") or "UNKNOWN"),
                    status=status,
                    severity=severity,
                    message=str(raw_check.get("message") or raw_check.get("code") or "check"),
                    details=details,
                )
            )
        return cls(
            status=str(report.get("status") or report.get("import_decision") or "unknown"),
            checks=tuple(checks),
            raw=dict(report),
        )


__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "ValidationSeverity",
    "ValidationStatus",
]
