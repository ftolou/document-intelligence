from __future__ import annotations

from pathlib import Path

import pytest

from receipt_intelligence.adapters.storage.sqlite.model_calls import SQLiteModelCallRepository
from receipt_intelligence.application.events import ModelCallCompletedEvent
from receipt_intelligence.application.ports.model_calls import ModelCallFilter
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def test_migration_seeds_current_luna_cache_aware_pricing(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "pricing.db")

    with database.connect() as connection:
        row = connection.execute(
            """
            SELECT provider, model, currency, input_price_per_million,
                   cached_input_price_per_million,
                   cache_write_input_price_per_million,
                   output_price_per_million, pricing_source, effective_from
            FROM model_pricing
            WHERE provider='openai' AND model='gpt-5.6-luna'
            """
        ).fetchone()

    assert row is not None
    assert row["currency"] == "USD"
    assert row["input_price_per_million"] == pytest.approx(0.20)
    assert row["cached_input_price_per_million"] == pytest.approx(0.02)
    assert row["cache_write_input_price_per_million"] == pytest.approx(0.25)
    assert row["output_price_per_million"] == pytest.approx(1.20)
    assert row["pricing_source"] == "openai_official_2026-07-30"
    assert row["effective_from"] == "2026-07-30"


def test_migration_repairs_legacy_display_name_and_pre_reduction_luna_price(
    tmp_path: Path,
) -> None:
    database = ReceiptDatabase(tmp_path / "legacy-pricing.db")
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM model_pricing WHERE provider='openai' AND model='gpt-5.6-luna'"
        )
        connection.execute(
            """
            INSERT INTO model_pricing(
                provider, model, currency, input_price_per_million,
                output_price_per_million, updated_at
            ) VALUES ('OpenAI', 'GPT-5.6 Luna', 'EUR', 1, 6, '2026-08-17T17:00:00Z')
            """
        )
        connection.execute("DELETE FROM schema_migrations WHERE version=10")
        connection.execute("UPDATE schema_meta SET value='9' WHERE key='schema_version'")
        connection.commit()

    database.migrations.migrate()

    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT provider, model, currency, input_price_per_million,
                   cached_input_price_per_million,
                   cache_write_input_price_per_million,
                   output_price_per_million
            FROM model_pricing
            WHERE lower(provider)='openai'
            """
        ).fetchall()

    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "openai"
    assert row["model"] == "gpt-5.6-luna"
    assert row["currency"] == "USD"
    assert row["input_price_per_million"] == pytest.approx(0.20)
    assert row["cached_input_price_per_million"] == pytest.approx(0.02)
    assert row["cache_write_input_price_per_million"] == pytest.approx(0.25)
    assert row["output_price_per_million"] == pytest.approx(1.20)


def test_openai_cost_uses_standard_cache_read_cache_write_and_output_rates(
    tmp_path: Path,
) -> None:
    database = ReceiptDatabase(tmp_path / "calls.db")
    repository = SQLiteModelCallRepository(database.db_path)
    repository.publish(
        ModelCallCompletedEvent(
            call_id="mc-openai-cost",
            occurred_at="2026-08-17T18:00:01Z",
            started_at="2026-08-17T18:00:00Z",
            trace_id="job-openai-cost",
            job_id="job-openai-cost",
            operation="receipt_extraction_one_shot",
            provider="openai",
            model="gpt-5.6-luna",
            endpoint="generate",
            status="completed",
            attempt=1,
            duration_ms=22_210.0,
            input_tokens=4092,
            output_tokens=2812,
            token_source="provider_reported",
            attributes={
                "cached_input_tokens": 2000,
                "cache_write_input_tokens": 1000,
                "reasoning_output_tokens": 1200,
            },
        )
    )

    calls = repository.list_calls(ModelCallFilter(), limit=10, offset=0)

    assert len(calls) == 1
    call = calls[0]
    assert call["standard_input_tokens"] == 1092
    assert call["cached_input_tokens"] == 2000
    assert call["cache_write_input_tokens"] == 1000
    assert call["reasoning_output_tokens"] == 1200
    expected = (1092 * 0.20 + 2000 * 0.02 + 1000 * 0.25 + 2812 * 1.20) / 1_000_000
    assert call["estimated_cost"] == pytest.approx(expected)
    assert call["missing_price_components"] == []


def test_model_catalog_exposes_canonical_ids_and_friendly_names(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "catalog.db")
    repository = SQLiteModelCallRepository(database.db_path)

    catalog = repository.list_models()

    luna = next(
        row for row in catalog if row["provider"] == "openai" and row["model"] == "gpt-5.6-luna"
    )
    assert luna["provider_display_name"] == "OpenAI"
    assert luna["model_display_name"] == "GPT-5.6 Luna"
    assert luna["display_name"] == "OpenAI — GPT-5.6 Luna"
    assert luna["has_pricing"] is True
