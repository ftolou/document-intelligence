"""Neutral event payloads emitted by application workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ExtractionRunEvent:
    """Snapshot of one extraction workflow execution."""

    event_name: ClassVar[str] = "extraction.run"
    schema_version: ClassVar[str] = "extraction_metrics_v2"

    run_id: str
    status: str
    started_at: str
    occurred_at: str
    duration_ms: float
    stages: tuple[Mapping[str, Any], ...]
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        stage_records = [dict(stage) for stage in self.stages]
        completed = sum(1 for stage in stage_records if stage.get("status") == "done")
        failed = sum(1 for stage in stage_records if stage.get("status") == "error")
        return {
            "schema_version": self.schema_version,
            "event_name": self.event_name,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.occurred_at,
            "duration_ms": round(float(self.duration_ms), 3),
            "stage_count": len(stage_records),
            "completed_stage_count": completed,
            "failed_stage_count": failed,
            "error": self.error,
            "stages": stage_records,
        }


@dataclass(frozen=True, slots=True)
class QueryExecutionEvent:
    """Compact transport-neutral snapshot of one query execution."""

    event_name: ClassVar[str] = "query.executed"
    schema_version: ClassVar[str] = "query_execution_event_v6"

    occurred_at: str
    query_id: str | None
    question: str | None
    engine: str
    engine_version: str | None
    orchestrator: str | None
    graph_version: str | None
    status: str | None
    duration_ms: float | None
    requested_api_limit: int | None
    stage_durations_ms: Mapping[str, float]
    validation_attempts: int
    repair_attempts: int
    row_count: int | None
    truncated: bool | None
    model_calls: Mapping[str, Any]
    errors: tuple[str, ...]
    error_code: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_name": self.event_name,
            "recorded_at": self.occurred_at,
            "query_id": self.query_id,
            "question": self.question,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "orchestrator": self.orchestrator,
            "graph_version": self.graph_version,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "requested_api_limit": self.requested_api_limit,
            "stage_durations_ms": dict(self.stage_durations_ms),
            "validation_attempts": self.validation_attempts,
            "repair_attempts": self.repair_attempts,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "model_calls": dict(self.model_calls),
            "errors": list(self.errors),
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class ModelCallCompletedEvent:
    """Provider-neutral record for one model request, successful or failed."""

    event_name: ClassVar[str] = "model.call.completed"
    schema_version: ClassVar[str] = "model_call_event_v2"

    call_id: str
    occurred_at: str
    started_at: str
    operation: str
    provider: str
    model: str | None
    endpoint: str
    status: str
    attempt: int
    duration_ms: float
    trace_id: str | None = None
    job_id: str | None = None
    receipt_id: str | None = None
    query_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_characters: int | None = None
    output_characters: int | None = None
    token_source: str = "unavailable"
    model_total_duration_ms: float | None = None
    model_load_duration_ms: float | None = None
    prompt_evaluation_duration_ms: float | None = None
    generation_duration_ms: float | None = None
    configured_context_window: int | None = None
    stop_reason: str | None = None
    error: str | None = None
    attributes: Mapping[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_name": self.event_name,
            "call_id": self.call_id,
            "recorded_at": self.occurred_at,
            "started_at": self.started_at,
            "trace_id": self.trace_id,
            "job_id": self.job_id,
            "receipt_id": self.receipt_id,
            "query_id": self.query_id,
            "operation": self.operation,
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "status": self.status,
            "attempt": self.attempt,
            "duration_ms": round(float(self.duration_ms), 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_characters": self.input_characters,
            "output_characters": self.output_characters,
            "token_source": self.token_source,
            "model_total_duration_ms": self.model_total_duration_ms,
            "model_load_duration_ms": self.model_load_duration_ms,
            "prompt_evaluation_duration_ms": self.prompt_evaluation_duration_ms,
            "generation_duration_ms": self.generation_duration_ms,
            "configured_context_window": self.configured_context_window,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "attributes": dict(self.attributes or {}),
        }


def query_execution_event_from_payload(
    payload: Mapping[str, Any],
    *,
    occurred_at: str,
) -> QueryExecutionEvent:
    execution = payload.get("execution") or {}
    diagnostics = payload.get("diagnostics") or {}
    stages = diagnostics.get("stages") or []
    stage_durations: dict[str, float] = {}
    validation_attempts = 0
    repair_attempts = 0
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("name") or "")
        duration = stage.get("duration_ms")
        if name and isinstance(duration, int | float):
            stage_durations[name] = float(duration)
        if name.startswith("validate_sql_attempt_"):
            validation_attempts += 1
        elif name.startswith("repair_sql_attempt_"):
            repair_attempts += 1

    data = payload.get("data") or {}
    model_calls = diagnostics.get("model_call_summary")
    if not isinstance(model_calls, dict):
        legacy_summary = diagnostics.get("ollama_summary")
        model_calls = legacy_summary if isinstance(legacy_summary, dict) else {}

    duration_ms = execution.get("duration_ms") or diagnostics.get("duration_ms")
    return QueryExecutionEvent(
        occurred_at=occurred_at,
        query_id=_optional_string(execution.get("query_id")),
        question=_optional_string(payload.get("question")),
        engine=str(execution.get("engine") or payload.get("strategy") or "rag_sql"),
        engine_version=_optional_string(
            execution.get("engine_version") or payload.get("engine_version")
        ),
        orchestrator=_optional_string(
            execution.get("orchestrator") or diagnostics.get("orchestrator")
        ),
        graph_version=_optional_string(
            execution.get("graph_version") or diagnostics.get("graph_version")
        ),
        status=_optional_string(execution.get("status") or payload.get("status")),
        duration_ms=float(duration_ms) if isinstance(duration_ms, int | float) else None,
        requested_api_limit=_optional_int(diagnostics.get("requested_api_limit")),
        stage_durations_ms=stage_durations,
        validation_attempts=validation_attempts,
        repair_attempts=repair_attempts,
        row_count=_optional_int(data.get("row_count")),
        truncated=data.get("truncated") if isinstance(data.get("truncated"), bool) else None,
        model_calls=model_calls,
        errors=normalize_errors(execution.get("errors")),
        error_code=_optional_string(payload.get("error_code")),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def normalize_errors(values: Sequence[object] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values if str(value).strip())


__all__ = [
    "ExtractionRunEvent",
    "ModelCallCompletedEvent",
    "QueryExecutionEvent",
    "query_execution_event_from_payload",
    "normalize_errors",
]
