"""Application-facing service for the single RAG-SQL query engine."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from receipt_intelligence.application.events import query_execution_event_from_payload
from receipt_intelligence.application.model_call_context import bind_model_call_context
from receipt_intelligence.application.ports.events import EventSink
from receipt_intelligence.observability.timing import utc_now_iso
from receipt_intelligence.rag_sql.orchestration.contracts import RAG_SQL_GRAPH_VERSION
from receipt_intelligence.rag_sql.runtime import (
    RagSqlRuntime,
    build_rag_sql_runtime_from_settings,
)


@dataclass(slots=True)
class ReceiptQueryService:
    """Execute receipt questions through RAG-SQL and attach app metadata."""

    runtime: RagSqlRuntime
    telemetry_sink: EventSink | None = None

    def execute(self, question: str, *, limit: int = 25) -> dict[str, Any]:
        query_id = f"q_{uuid.uuid4().hex}"
        started_at = utc_now_iso()
        started_perf = time.perf_counter()

        with bind_model_call_context(trace_id=query_id, query_id=query_id):
            response = self.runtime.execute(question)
        payload = response.model_dump(mode="json")
        diagnostics = payload.setdefault("diagnostics", {})
        orchestrator_name = str(
            getattr(self.runtime, "orchestrator_name", None)
            or diagnostics.get("orchestrator")
            or "langgraph"
        )
        orchestrator_version = str(
            getattr(self.runtime, "orchestrator_version", None)
            or diagnostics.get("graph_version")
            or RAG_SQL_GRAPH_VERSION
        )
        diagnostics.setdefault("requested_api_limit", max(1, min(100, int(limit))))
        diagnostics.setdefault("orchestrator", orchestrator_name)
        diagnostics.setdefault("graph_version", orchestrator_version)
        duration_ms = diagnostics.get("duration_ms")
        if not isinstance(duration_ms, int | float):
            duration_ms = (time.perf_counter() - started_perf) * 1000.0
            diagnostics["duration_ms"] = duration_ms

        errors: list[str] = []
        if payload.get("error"):
            errors.append(str(payload["error"]))

        payload["execution"] = {
            "engine": "rag_sql",
            "engine_version": payload.get("engine_version"),
            "orchestrator": orchestrator_name,
            "graph_version": orchestrator_version,
            "query_id": query_id,
            "started_at": started_at,
            "duration_ms": float(duration_ms),
            "status": payload.get("status"),
            "errors": errors,
            "financial_calculation": "deterministic_sql",
            "sql_generation_by_llm": True,
        }
        if self.telemetry_sink is not None:
            self.telemetry_sink.publish(
                query_execution_event_from_payload(payload, occurred_at=utc_now_iso())
            )
        return payload

    def close(self) -> None:
        """Release process-scoped query resources during application shutdown."""

        self.runtime.close()


def build_receipt_query_service_from_settings(
    *,
    telemetry_sink: EventSink | None = None,
    model_call_sink: EventSink | None = None,
) -> ReceiptQueryService:
    return ReceiptQueryService(
        runtime=build_rag_sql_runtime_from_settings(event_sink=model_call_sink),
        telemetry_sink=telemetry_sink,
    )


__all__ = ["ReceiptQueryService", "build_receipt_query_service_from_settings"]
