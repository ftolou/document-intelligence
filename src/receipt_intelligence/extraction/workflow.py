"""Sequential staged workflow for receipt extraction."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import replace

from receipt_intelligence.application.events import ExtractionRunEvent
from receipt_intelligence.extraction.artifacts import copy_alias, save_json
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.stages.base import ExtractionStage
from receipt_intelligence.extraction.state import ExtractionPhase
from receipt_intelligence.observability.timing import elapsed_ms, utc_now_iso


class ReceiptExtractionWorkflow:
    def __init__(self, stages: Iterable[ExtractionStage]) -> None:
        self.stages = tuple(stages)
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Extraction stage names must be unique.")

    def run(self, context: ExtractionContext) -> ExtractionContext:
        for stage in self.stages:
            input_phase = getattr(stage, "input_phase", None)
            output_phase = getattr(stage, "output_phase", None)
            if input_phase is not None:
                context.assert_phase(input_phase, stage.name)

            started = time.perf_counter()
            trace = {
                "stage": stage.name,
                "status": "running",
                "started_at": utc_now_iso(),
                "input_phase": _phase_value(input_phase),
                "output_phase": _phase_value(output_phase),
            }
            context.stage_trace.append(trace)
            try:
                updated = stage.run(context)
                if updated is not context:
                    context = updated
                if input_phase is not None and output_phase is not None:
                    context.advance_phase(input_phase, output_phase, stage.name)
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

        _refresh_finalization_observability(context)
        self._persist_observability(context, status="completed")
        return context

    @staticmethod
    def _persist_observability(
        context: ExtractionContext,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        paths = context.available_paths
        stage_trace_path = paths.get("stage_trace")
        if stage_trace_path:
            save_json(stage_trace_path, context.stage_trace)
            stage_trace_alias = context.config.result_dir / "latest_extraction_stage_trace.json"
            copy_alias(stage_trace_path, stage_trace_alias)
            paths["latest_extraction_stage_trace"] = stage_trace_alias
        metrics_path = paths.get("extraction_metrics")
        if metrics_path:
            metrics_alias = context.config.result_dir / "latest_extraction_metrics.json"
            paths["latest_extraction_metrics"] = metrics_alias
            context.dependencies.event_sink.publish(
                ExtractionRunEvent(
                    run_id=context.config.run_id,
                    status=status,
                    started_at=context.started_at_utc,
                    occurred_at=utc_now_iso(),
                    duration_ms=round(context.duration_seconds * 1000.0, 3),
                    stages=tuple(dict(entry) for entry in context.stage_trace),
                    error=error,
                )
            )


def _phase_value(phase: ExtractionPhase | None) -> str | None:
    return phase.value if phase is not None else None


def _refresh_finalization_observability(context: ExtractionContext) -> None:
    """Replace the in-stage metadata snapshot with the completed workflow trace."""

    finalization = context.finalized
    if finalization is None or finalization.next_finalization is None:
        return
    result = finalization.next_finalization
    metadata = dict(result.pipeline_metadata)
    workflow = dict(metadata.get("workflow") or {})
    workflow["stage_trace"] = [dict(entry) for entry in context.stage_trace]
    workflow["stage_count"] = len(context.stage_trace)
    workflow["stages"] = [str(entry.get("stage") or "") for entry in context.stage_trace]
    metadata["workflow"] = workflow
    updated = replace(result, pipeline_metadata=metadata)
    finalization.next_finalization = updated
    finalization.pipeline_meta = metadata
    for key in ("pipeline_meta", "latest_pipeline_meta"):
        path = context.available_paths.get(key)
        if path is not None:
            save_json(path, metadata)
