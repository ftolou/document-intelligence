"""Ask-your-receipts application use case."""

from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from receipt_intelligence.application.errors import InvalidRequestError
from receipt_intelligence.application.query_diagnostics import capture_query_diagnostics

_ALLOWED_FIELDS = {"question", "limit", "save_json_log"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


class ReceiptQueryExecutor(Protocol):
    def execute(self, question: str, *, limit: int = 25) -> dict[str, Any]: ...

    def close(self) -> None: ...


class ReceiptQueryLogWriter(Protocol):
    def write(self, record: Mapping[str, Any], *, log_id: str) -> str | None: ...


class AskReceipts:
    def __init__(
        self,
        query_service: ReceiptQueryExecutor,
        *,
        log_writer: ReceiptQueryLogWriter | None = None,
    ) -> None:
        self._query_service = query_service
        self._log_writer = log_writer

    def execute(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise InvalidRequestError("Expected JSON object.", code="invalid_request")
        unsupported_fields = sorted(set(payload) - _ALLOWED_FIELDS)
        if unsupported_fields:
            raise InvalidRequestError(
                f"Unsupported request field(s): {', '.join(unsupported_fields)}.",
                code="unsupported_request_field",
            )
        question = str(payload.get("question") or "").strip()
        if not question:
            raise InvalidRequestError("Missing question.", code="missing_question")
        try:
            limit = max(1, min(100, int(payload.get("limit", 25))))
        except (TypeError, ValueError):
            limit = 25
        save_json_log = _parse_boolean(payload.get("save_json_log", False))

        request_id = f"ask_{uuid.uuid4().hex}"
        started_at = _utc_now_iso()
        started_perf = time.perf_counter()

        with capture_query_diagnostics(enabled=save_json_log) as diagnostics:
            try:
                result = dict(self._query_service.execute(question, limit=limit))
            except Exception as exc:
                metadata = self._write_log(
                    log_id=request_id,
                    record={
                        "schema_version": "ask_receipts_diagnostic_v1",
                        "request_id": request_id,
                        "query_id": None,
                        "started_at": started_at,
                        "completed_at": _utc_now_iso(),
                        "duration_ms": round((time.perf_counter() - started_perf) * 1000.0, 3),
                        "status": "error",
                        "request": {
                            "question": question,
                            "limit": limit,
                        },
                        "response": None,
                        "exception": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "error_code": getattr(exc, "code", None),
                            "traceback": "".join(
                                traceback.format_exception(type(exc), exc, exc.__traceback__)
                            ),
                        },
                        "diagnostic_events": diagnostics.snapshot(),
                    },
                    enabled=save_json_log,
                )
                _attach_log_metadata(exc, metadata)
                raise

            query_id = _query_id_from_result(result)
            metadata = self._write_log(
                log_id=query_id or request_id,
                record={
                    "schema_version": "ask_receipts_diagnostic_v1",
                    "request_id": request_id,
                    "query_id": query_id,
                    "started_at": started_at,
                    "completed_at": _utc_now_iso(),
                    "duration_ms": round((time.perf_counter() - started_perf) * 1000.0, 3),
                    "status": str(result.get("status") or "completed"),
                    "request": {
                        "question": question,
                        "limit": limit,
                    },
                    "response": result,
                    "exception": None,
                    "diagnostic_events": diagnostics.snapshot(),
                },
                enabled=save_json_log,
            )
            if metadata is not None:
                result["diagnostic_log"] = metadata
            return result

    def _write_log(
        self,
        *,
        log_id: str,
        record: Mapping[str, Any],
        enabled: bool,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        filename = None
        if self._log_writer is not None:
            try:
                filename = self._log_writer.write(record, log_id=log_id)
            except Exception:
                filename = None
        return {
            "enabled": True,
            "saved": filename is not None,
            "filename": filename,
        }


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise InvalidRequestError(
        "save_json_log must be a boolean.",
        code="invalid_request",
    )


def _query_id_from_result(result: Mapping[str, Any]) -> str | None:
    execution = result.get("execution")
    if not isinstance(execution, Mapping):
        return None
    value = str(execution.get("query_id") or "").strip()
    return value or None


def _attach_log_metadata(exc: Exception, metadata: dict[str, Any] | None) -> None:
    if metadata is None:
        return
    try:
        exc.diagnostic_log = metadata
    except Exception:
        return


__all__ = ["AskReceipts", "ReceiptQueryExecutor", "ReceiptQueryLogWriter"]
