"""Compact telemetry for RAG-SQL LangGraph executions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from receipt_intelligence.observability.jsonl import JsonlEventWriter
from receipt_intelligence.observability.timing import utc_now_iso


@dataclass(slots=True)
class QueryTelemetrySink:
    """Persist compact query execution events to a JSONL file."""

    writer: JsonlEventWriter | None

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        enabled: bool = True,
    ) -> QueryTelemetrySink:
        return cls(JsonlEventWriter(path) if enabled else None)

    def record(self, response: dict[str, Any]) -> None:
        if self.writer is None:
            return

        execution = response.get("execution") or {}
        diagnostics = response.get("diagnostics") or {}
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

        data = response.get("data") or {}
        ollama = diagnostics.get("ollama_summary") or {}
        event = {
            "schema_version": "query_execution_event_v5",
            "recorded_at": utc_now_iso(),
            "query_id": execution.get("query_id"),
            "question": response.get("question"),
            "engine": execution.get("engine") or response.get("strategy") or "rag_sql",
            "engine_version": execution.get("engine_version") or response.get("engine_version"),
            "orchestrator": execution.get("orchestrator") or diagnostics.get("orchestrator"),
            "graph_version": execution.get("graph_version") or diagnostics.get("graph_version"),
            "status": execution.get("status") or response.get("status"),
            "duration_ms": execution.get("duration_ms") or diagnostics.get("duration_ms"),
            "requested_api_limit": diagnostics.get("requested_api_limit"),
            "stage_durations_ms": stage_durations,
            "validation_attempts": validation_attempts,
            "repair_attempts": repair_attempts,
            "row_count": data.get("row_count"),
            "truncated": data.get("truncated"),
            "ollama": {
                "call_count": ollama.get("call_count"),
                "total_request_duration_ms": ollama.get("total_request_duration_ms"),
                "total_load_duration_ms": ollama.get("total_load_duration_ms"),
                "total_generation_duration_ms": ollama.get("total_generation_duration_ms"),
            },
            "errors": execution.get("errors") or [],
            "error_code": response.get("error_code"),
        }
        try:
            self.writer.append(event)
        except OSError:
            return
