"""Compatibility observability exports and lightweight timing helpers.

New workflows publish typed application events through ``EventSink`` ports.
Concrete persistence implementations live under ``adapters.observability``.
"""

from receipt_intelligence.application.ports.llm import ModelCallMetrics
from receipt_intelligence.observability.jsonl import JsonlEventWriter
from receipt_intelligence.observability.query import QueryTelemetrySink
from receipt_intelligence.observability.readiness import build_readiness_report
from receipt_intelligence.observability.timing import elapsed_ms, utc_now_iso

__all__ = [
    "JsonlEventWriter",
    "ModelCallMetrics",
    "QueryTelemetrySink",
    "build_readiness_report",
    "elapsed_ms",
    "utc_now_iso",
]
