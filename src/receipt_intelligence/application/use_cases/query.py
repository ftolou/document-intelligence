"""Ask-your-receipts application use case."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from receipt_intelligence.application.errors import InvalidRequestError

_ALLOWED_FIELDS = {"question", "limit"}


class ReceiptQueryExecutor(Protocol):
    def execute(self, question: str, *, limit: int = 25) -> dict[str, Any]: ...

    def close(self) -> None: ...


class AskReceipts:
    def __init__(self, query_service: ReceiptQueryExecutor) -> None:
        self._query_service = query_service

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
        return self._query_service.execute(question, limit=limit)


__all__ = ["AskReceipts", "ReceiptQueryExecutor"]
