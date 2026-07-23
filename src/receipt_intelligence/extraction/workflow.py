"""Sequential staged workflow for receipt extraction."""

from __future__ import annotations

import time
from collections.abc import Iterable

from receipt_intelligence.extraction.artifacts import copy_alias, save_json
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.stages.base import ExtractionStage
from receipt_intelligence.observability.extraction import build_extraction_metrics
from receipt_intelligence.observability.timing import elapsed_ms, utc_now_iso


class ReceiptExtractionWorkflow:
    def __init__(self, stages: Iterable[ExtractionStage]) -> None:
        self.stages = tuple(stages)
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Extraction stage names must be unique.")

    def run(self, context: ExtractionContext) -> ExtractionContext:
        for stage in self.stages:
            started = time.perf_counter()
            trace = {
                "stage": stage.name,
                "status": "running",
                "started_at": utc_now_iso(),
            }
            context.stage_trace.append(trace)
            try:
                updated = stage.run(context)
                if updated is not context:
                    context = updated
                trace.update(
                    status="done",
                    finished_at=utc_now_iso(),
                    duration_ms=elapsed_ms(started),
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                trace.update(
                    status="error",
                    finished_at=utc_now_iso(),
                    duration_ms=elapsed_ms(started),
                    error=error,
                )
                self._persist_observability(context, status="error", error=error)
                raise
            self._persist_observability(context, status="running")

        self._persist_observability(context, status="completed")
        return context

    @staticmethod
    def _persist_observability(
        context: ExtractionContext,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        stage_trace_path = context.paths.get("stage_trace")
        if stage_trace_path:
            save_json(stage_trace_path, context.stage_trace)
            stage_trace_alias = context.config.result_dir / "latest_extraction_stage_trace.json"
            copy_alias(stage_trace_path, stage_trace_alias)
            context.paths["latest_extraction_stage_trace"] = stage_trace_alias
        metrics_path = context.paths.get("extraction_metrics")
        if metrics_path:
            save_json(
                metrics_path,
                build_extraction_metrics(context, status=status, error=error),
            )
            metrics_alias = context.config.result_dir / "latest_extraction_metrics.json"
            copy_alias(metrics_path, metrics_alias)
            context.paths["latest_extraction_metrics"] = metrics_alias
