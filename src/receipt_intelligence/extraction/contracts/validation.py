"""Typed read-only deterministic validation contracts."""

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
    REVIEW = "review"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ValidationRequest:
    receipt: JsonObject
    item_contract: JsonObject
    item_pipeline_enabled: bool
    selected_scalar_tasks: tuple[str, ...]
    money_tolerance: float = 0.02
    vat_rate_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, dict):
            raise TypeError("ValidationRequest.receipt must be an object.")
        if not isinstance(self.item_contract, dict):
            raise TypeError("ValidationRequest.item_contract must be an object.")
        if self.money_tolerance < 0 or self.vat_rate_tolerance < 0:
            raise ValueError("Validation tolerances must be nonnegative.")
        object.__setattr__(self, "receipt", dict(self.receipt))
        object.__setattr__(self, "item_contract", dict(self.item_contract))
        object.__setattr__(
            self,
            "selected_scalar_tasks",
            tuple(str(value).strip() for value in self.selected_scalar_tasks if str(value).strip()),
        )


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    code: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    values: JsonObject = field(default_factory=dict)
    details: Any = None

    def __post_init__(self) -> None:
        code = str(self.code or "").strip()
        message = str(self.message or "").strip()
        if not code:
            raise ValueError("ValidationCheck.code must not be empty.")
        if not message:
            raise ValueError("ValidationCheck.message must not be empty.")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "values", dict(self.values))

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "code": self.code,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.values:
            payload["values"] = dict(self.values)
        if self.details is not None:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True, slots=True)
class ValidationReport:
    status: str
    checks: tuple[ValidationCheck, ...]
    raw: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status or "").strip()
        if not status:
            raise ValueError("ValidationReport.status must not be empty.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "raw", dict(self.raw))

    @property
    def failed_codes(self) -> frozenset[str]:
        return frozenset(
            check.code for check in self.checks if check.status is ValidationStatus.FAILED
        )

    def find(self, code: str) -> ValidationCheck | None:
        return next((check for check in self.checks if check.code == code), None)

    def to_dict(self) -> JsonObject:
        if self.raw:
            return dict(self.raw)
        return {"status": self.status, "checks": [check.to_dict() for check in self.checks]}

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
            values = raw_check.get("values") if isinstance(raw_check.get("values"), dict) else {}
            checks.append(
                ValidationCheck(
                    code=str(raw_check.get("code") or "UNKNOWN"),
                    status=status,
                    severity=severity,
                    message=str(raw_check.get("message") or raw_check.get("code") or "check"),
                    values=values,
                    details=raw_check.get("details"),
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
    "ValidationRequest",
    "ValidationSeverity",
    "ValidationStatus",
]
