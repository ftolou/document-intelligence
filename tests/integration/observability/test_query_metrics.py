from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from receipt_intelligence.observability.query import QueryTelemetrySink
from receipt_intelligence.rag_sql.application import ReceiptQueryService
from receipt_intelligence.rag_sql.models import RagSqlResponse, SqlExecutionResult


@dataclass
class _StubRuntime:
    def execute(self, question: str) -> RagSqlResponse:
        return RagSqlResponse(
            question=question,
            status="completed",
            answer="20.00 EUR",
            data=SqlExecutionResult(
                columns=["value", "currency"],
                rows=[{"value": 20.0, "currency": "EUR"}],
                row_count=1,
            ),
            diagnostics={
                "orchestrator": "langgraph",
                "graph_version": "rag_sql_graph_v2",
                "duration_ms": 8.5,
                "stages": [
                    {"name": "analyze_question", "status": "done", "duration_ms": 2.0},
                    {"name": "generate_sql", "status": "done", "duration_ms": 3.0},
                    {"name": "validate_sql_attempt_1", "status": "done", "duration_ms": 0.5},
                    {"name": "execute_sql", "status": "done", "duration_ms": 1.0},
                ],
                "ollama_summary": {"call_count": 2},
            },
        )


def test_application_service_exposes_and_persists_graph_metrics(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "query_events.jsonl"
    service = ReceiptQueryService(
        runtime=_StubRuntime(),  # type: ignore[arg-type]
        telemetry_sink=QueryTelemetrySink.from_path(telemetry_path),
    )

    result = service.execute("How much did I spend at REWE?", limit=25)

    execution = result["execution"]
    assert execution["query_id"].startswith("q_")
    assert execution["engine"] == "rag_sql"
    assert execution["orchestrator"] == "langgraph"
    assert execution["graph_version"] == "rag_sql_graph_v2"
    assert execution["duration_ms"] >= 0
    event = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert event["query_id"] == execution["query_id"]
    assert event["engine"] == "rag_sql"
    assert event["orchestrator"] == "langgraph"
    assert event["row_count"] == 1
