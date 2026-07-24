"""Compatibility helpers for extraction event serialization.

New extraction code emits :class:`application.events.ExtractionRunEvent` through
an application ``EventSink``. This module remains for callers that need the
historical metrics dictionary without importing extraction workflow types.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from receipt_intelligence.application.events import ExtractionRunEvent
from receipt_intelligence.observability.timing import utc_now_iso


def build_extraction_metrics(
    context: object | None = None,
    *,
    run_id: str | None = None,
    status: str,
    started_at: str | None = None,
    updated_at: str | None = None,
    duration_ms: float | None = None,
    stages: Sequence[Mapping[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Serialize extraction metrics without importing ``ExtractionContext``.

    Passing ``context`` is supported for historical callers through structural
    attribute access. New callers should provide the explicit neutral fields.
    """

    if context is not None:
        config = context.config
        run_id = str(config.run_id)
        started_at = str(context.started_at_utc)
        duration_ms = float(context.duration_seconds) * 1000.0
        stages = tuple(context.stage_trace)

    if run_id is None or started_at is None or duration_ms is None or stages is None:
        raise TypeError(
            "build_extraction_metrics requires either context or explicit run fields."
        )

    return ExtractionRunEvent(
        run_id=run_id,
        status=status,
        started_at=started_at,
        occurred_at=updated_at or utc_now_iso(),
        duration_ms=duration_ms,
        stages=tuple(stages),
        error=error,
    ).to_record()


__all__ = ["build_extraction_metrics"]
