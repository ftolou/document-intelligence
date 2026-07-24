from __future__ import annotations

from receipt_intelligence.application.events import query_execution_event_from_payload


def test_query_event_accepts_historical_provider_summary() -> None:
    event = query_execution_event_from_payload(
        {
            "question": "Total?",
            "status": "completed",
            "data": {"row_count": 1, "truncated": False},
            "diagnostics": {
                "duration_ms": 5.0,
                "ollama_summary": {"call_count": 1},
                "stages": [],
            },
            "execution": {"query_id": "q_1", "errors": []},
        },
        occurred_at="2026-07-24T12:00:00.000Z",
    )

    record = event.to_record()
    assert record["model_calls"] == {"call_count": 1}
    assert "ollama" not in record
