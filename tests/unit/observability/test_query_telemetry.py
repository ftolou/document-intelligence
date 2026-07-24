from __future__ import annotations

import json
from pathlib import Path

from receipt_intelligence.observability.query import QueryTelemetrySink


def test_query_telemetry_writes_provider_neutral_graph_event(tmp_path: Path) -> None:
    path = tmp_path / "query_events.jsonl"
    sink = QueryTelemetrySink.from_path(path)
    sink.record(
        {
            "strategy": "rag_sql",
            "engine_version": "rag_sql_engine_v2",
            "question": "How much at REWE?",
            "status": "completed",
            "data": {"row_count": 1, "truncated": False},
            "diagnostics": {
                "orchestrator": "langgraph",
                "graph_version": "rag_sql_graph_v2",
                "requested_api_limit": 25,
                "duration_ms": 12.5,
                "stages": [
                    {"name": "analyze_question", "status": "done", "duration_ms": 2.0},
                    {"name": "validate_sql_attempt_1", "status": "done", "duration_ms": 0.5},
                    {"name": "execute_sql", "status": "done", "duration_ms": 1.0},
                ],
                "model_call_summary": {
                    "call_count": 2,
                    "providers": {"ollama": 2},
                    "total_request_duration_ms": 10.0,
                    "total_load_duration_ms": 1.0,
                    "total_generation_duration_ms": 7.0,
                },
            },
            "execution": {
                "query_id": "q_test",
                "engine": "rag_sql",
                "engine_version": "rag_sql_engine_v2",
                "orchestrator": "langgraph",
                "graph_version": "rag_sql_graph_v2",
                "status": "completed",
                "duration_ms": 12.5,
                "errors": [],
            },
        }
    )

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["schema_version"] == "query_execution_event_v6"
    assert event["event_name"] == "query.executed"
    assert event["query_id"] == "q_test"
    assert event["engine"] == "rag_sql"
    assert event["orchestrator"] == "langgraph"
    assert event["graph_version"] == "rag_sql_graph_v2"
    assert event["validation_attempts"] == 1
    assert event["repair_attempts"] == 0
    assert event["row_count"] == 1
    assert event["model_calls"]["call_count"] == 2
    assert event["model_calls"]["providers"] == {"ollama": 2}
    assert "ollama" not in event
    assert "tool_calls" not in event
