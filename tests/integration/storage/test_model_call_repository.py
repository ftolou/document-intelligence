from __future__ import annotations

from receipt_intelligence.adapters.storage.sqlite.model_calls import (
    SQLiteModelCallRepository,
)
from receipt_intelligence.application.events import ModelCallCompletedEvent
from receipt_intelligence.application.ports.model_calls import (
    ModelCallFilter,
    ModelPricingInput,
)
from receipt_intelligence.storage.bootstrap import initialize_database


def test_model_call_repository_calculates_configured_token_cost(tmp_path) -> None:
    database_path = tmp_path / "receipt.db"
    initialize_database(database_path)
    repository = SQLiteModelCallRepository(database_path)
    repository.upsert_pricing(
        ModelPricingInput(
            provider="ollama",
            model="gemma4",
            currency="EUR",
            input_price_per_million=2.0,
            output_price_per_million=8.0,
        )
    )
    repository.publish(
        ModelCallCompletedEvent(
            call_id="mc_1",
            occurred_at="2026-07-24T12:00:01.000Z",
            started_at="2026-07-24T12:00:00.000Z",
            trace_id="job_1",
            job_id="job_1",
            operation="receipt_main_parse",
            provider="ollama",
            model="gemma4",
            endpoint="generate",
            status="completed",
            attempt=1,
            duration_ms=3000.0,
            input_tokens=1000,
            output_tokens=250,
            prompt_evaluation_duration_ms=1000.0,
            generation_duration_ms=2000.0,
            token_source="provider_reported",
        )
    )

    calls = repository.list_calls(ModelCallFilter(), limit=10, offset=0)
    summary = repository.summary(ModelCallFilter())

    assert calls[0]["estimated_cost"] == 0.004
    assert calls[0]["generated_tokens_per_second"] == 125.0
    assert summary["input_tokens"] == 1000
    assert summary["output_tokens"] == 250
    assert summary["estimated_cost"] == 0.004
    assert summary["currency"] == "EUR"


def test_model_call_repository_marks_calls_without_pricing(tmp_path) -> None:
    database_path = tmp_path / "receipt.db"
    initialize_database(database_path)
    repository = SQLiteModelCallRepository(database_path)
    repository.publish(
        ModelCallCompletedEvent(
            call_id="mc_2",
            occurred_at="2026-07-24T12:00:01.000Z",
            started_at="2026-07-24T12:00:00.000Z",
            operation="rag_sql_planning",
            provider="ollama",
            model="qwen",
            endpoint="generate",
            status="completed",
            attempt=1,
            duration_ms=100.0,
            input_tokens=50,
            output_tokens=10,
        )
    )

    summary = repository.summary(ModelCallFilter())
    assert summary["estimated_cost"] is None
    assert summary["unpriced_call_count"] == 1
