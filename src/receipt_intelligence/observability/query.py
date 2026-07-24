"""Compatibility adapter for historical query telemetry imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from receipt_intelligence.adapters.observability import JsonlEventSink
from receipt_intelligence.application.events import query_execution_event_from_payload
from receipt_intelligence.observability.timing import utc_now_iso


class QueryTelemetrySink(JsonlEventSink):
    """Deprecated JSONL sink kept for callers using the historical class name.

    New application composition should inject ``EventSink`` and publish typed
    application events. ``record`` remains only as a compatibility edge.
    """

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        enabled: bool = True,
    ) -> QueryTelemetrySink:
        return cls(path, enabled=enabled)

    def record(self, response: dict[str, Any]) -> None:
        self.publish(
            query_execution_event_from_payload(response, occurred_at=utc_now_iso())
        )


__all__ = ["QueryTelemetrySink"]
