"""Extraction-stage telemetry summaries."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.observability.timing import utc_now_iso


def build_extraction_metrics(
    context: ExtractionContext,
    *,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    stages = [dict(entry) for entry in context.stage_trace]
    completed = sum(1 for entry in stages if entry.get("status") == "done")
    failed = sum(1 for entry in stages if entry.get("status") == "error")
    return {
        "schema_version": "extraction_metrics_v1",
        "run_id": context.config.run_id,
        "status": status,
        "started_at": context.started_at_utc,
        "updated_at": utc_now_iso(),
        "duration_ms": round(context.duration_seconds * 1000.0, 3),
        "stage_count": len(stages),
        "completed_stage_count": completed,
        "failed_stage_count": failed,
        "error": error,
        "stages": stages,
    }
