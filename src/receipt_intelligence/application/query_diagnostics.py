"""Request-scoped diagnostic capture for opt-in Ask Your Receipts logging."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class QueryDiagnosticCapture:
    """Collect detailed diagnostics only for the current query request."""

    enabled: bool = False
    _entries: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, event: str, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        entry = {
            "recorded_at": _utc_now_iso(),
            "event": str(event or "query.diagnostic"),
            **dict(payload),
        }
        with self._lock:
            self._entries.append(entry)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(entry) for entry in self._entries]


_CURRENT_QUERY_DIAGNOSTICS: ContextVar[QueryDiagnosticCapture | None] = ContextVar(
    "receipt_intelligence_query_diagnostics",
    default=None,
)


def current_query_diagnostics() -> QueryDiagnosticCapture | None:
    return _CURRENT_QUERY_DIAGNOSTICS.get()


def record_query_diagnostic(event: str, payload: Mapping[str, Any]) -> None:
    capture = current_query_diagnostics()
    if capture is not None:
        capture.record(event, payload)


@contextmanager
def capture_query_diagnostics(*, enabled: bool) -> Iterator[QueryDiagnosticCapture]:
    capture = QueryDiagnosticCapture(enabled=bool(enabled))
    token: Token[QueryDiagnosticCapture | None] = _CURRENT_QUERY_DIAGNOSTICS.set(capture)
    try:
        yield capture
    finally:
        _CURRENT_QUERY_DIAGNOSTICS.reset(token)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "QueryDiagnosticCapture",
    "capture_query_diagnostics",
    "current_query_diagnostics",
    "record_query_diagnostic",
]
