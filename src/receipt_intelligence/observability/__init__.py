"""Application-owned observability helpers.

The package intentionally avoids external telemetry dependencies. Runtime events
are represented as JSON-compatible dictionaries and can be persisted as JSONL.
"""

from receipt_intelligence.observability.jsonl import JsonlEventWriter
from receipt_intelligence.observability.ollama import OllamaCallMetrics
from receipt_intelligence.observability.query import QueryTelemetrySink
from receipt_intelligence.observability.readiness import build_readiness_report
from receipt_intelligence.observability.timing import elapsed_ms, utc_now_iso

__all__ = [
    "JsonlEventWriter",
    "OllamaCallMetrics",
    "QueryTelemetrySink",
    "build_readiness_report",
    "elapsed_ms",
    "utc_now_iso",
]
