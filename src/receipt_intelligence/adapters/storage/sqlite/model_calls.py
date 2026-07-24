"""SQLite read model and event sink for model-call telemetry."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from receipt_intelligence.application.ports.events import ApplicationEvent
from receipt_intelligence.application.ports.model_calls import (
    ModelCallFilter,
    ModelPricingInput,
)
from receipt_intelligence.observability.timing import utc_now_iso
from receipt_intelligence.storage.connection import SQLiteConnectionFactory


class SQLiteModelCallRepository:
    """Persist model events and serve dashboard-oriented aggregations."""

    def __init__(self, database_path: Path | str, *, enabled: bool = True) -> None:
        self.connections = SQLiteConnectionFactory(database_path)
        self.enabled = bool(enabled)

    def publish(self, event: ApplicationEvent) -> None:
        if not self.enabled or event.event_name != "model.call.completed":
            return
        record = event.to_record()
        try:
            with self.connections.connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO model_calls(
                        call_id, recorded_at, started_at, trace_id, job_id, receipt_id,
                        query_id, operation, provider, model, endpoint, status, attempt,
                        duration_ms, input_tokens, output_tokens, input_characters,
                        output_characters, token_source, model_total_duration_ms,
                        model_load_duration_ms, prompt_evaluation_duration_ms,
                        generation_duration_ms, configured_context_window, stop_reason,
                        error, attributes_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        record["call_id"], record["recorded_at"], record["started_at"],
                        record.get("trace_id"), record.get("job_id"),
                        record.get("receipt_id"), record.get("query_id"),
                        record["operation"], record["provider"], record.get("model"),
                        record["endpoint"], record["status"],
                        int(record.get("attempt") or 1),
                        float(record.get("duration_ms") or 0.0),
                        record.get("input_tokens"), record.get("output_tokens"),
                        record.get("input_characters"), record.get("output_characters"),
                        record.get("token_source") or "unavailable",
                        record.get("model_total_duration_ms"),
                        record.get("model_load_duration_ms"),
                        record.get("prompt_evaluation_duration_ms"),
                        record.get("generation_duration_ms"),
                        record.get("configured_context_window"),
                        record.get("stop_reason"), record.get("error"),
                        json.dumps(record.get("attributes") or {}, ensure_ascii=False),
                    ),
                )
                connection.commit()
        except sqlite3.Error:
            # Telemetry must never make a successful model call fail. Startup
            # migrations create the table for the normal web application path.
            return

    def summary(self, filters: ModelCallFilter) -> dict[str, Any]:
        where_sql, parameters = _where_clause(filters)
        with self.connections.connect_read_only() as connection:
            rows = connection.execute(
                f"""
                SELECT mc.*, p.currency, p.input_price_per_million,
                       p.output_price_per_million
                FROM model_calls mc
                LEFT JOIN model_pricing p
                  ON p.provider = mc.provider AND p.model = COALESCE(mc.model, '')
                {where_sql}
                ORDER BY mc.recorded_at DESC
                """,
                parameters,
            ).fetchall()

        records = [_row_to_call(row) for row in rows]
        durations = sorted(float(record["duration_ms"]) for record in records)
        priced_costs = [record["estimated_cost"] for record in records if record["estimated_cost"] is not None]
        total_input = sum(int(record["input_tokens"] or 0) for record in records)
        total_output = sum(int(record["output_tokens"] or 0) for record in records)
        completed = sum(1 for record in records if record["status"] == "completed")
        failed = sum(1 for record in records if record["status"] == "failed")
        unpriced = sum(
            1
            for record in records
            if record["estimated_cost"] is None
            and ((record["input_tokens"] or 0) > 0 or (record["output_tokens"] or 0) > 0)
        )
        return {
            "call_count": len(records),
            "completed_call_count": completed,
            "failed_call_count": failed,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "average_duration_ms": _average(durations),
            "p95_duration_ms": _percentile(durations, 0.95),
            "average_generated_tokens_per_second": _average(
                [
                    float(record["generated_tokens_per_second"])
                    for record in records
                    if record["generated_tokens_per_second"] is not None
                ]
            ),
            "estimated_cost": (
                round(sum(float(value) for value in priced_costs), 8)
                if priced_costs
                else None
            ),
            "priced_call_count": len(priced_costs),
            "unpriced_call_count": unpriced,
            "currency": _single_currency(records),
            "by_operation": _group_records(records, "operation"),
            "by_model": _group_records(records, "model_key"),
        }

    def list_calls(
        self,
        filters: ModelCallFilter,
        *,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        where_sql, parameters = _where_clause(filters)
        with self.connections.connect_read_only() as connection:
            rows = connection.execute(
                f"""
                SELECT mc.*, p.currency, p.input_price_per_million,
                       p.output_price_per_million
                FROM model_calls mc
                LEFT JOIN model_pricing p
                  ON p.provider = mc.provider AND p.model = COALESCE(mc.model, '')
                {where_sql}
                ORDER BY mc.recorded_at DESC
                LIMIT ? OFFSET ?
                """,
                (*parameters, max(1, min(500, int(limit))), max(0, int(offset))),
            ).fetchall()
        return [_row_to_call(row) for row in rows]

    def list_pricing(self) -> list[dict[str, Any]]:
        with self.connections.connect_read_only() as connection:
            rows = connection.execute(
                """
                SELECT provider, model, currency, input_price_per_million,
                       output_price_per_million, updated_at
                FROM model_pricing
                ORDER BY provider, model
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_pricing(self, pricing: ModelPricingInput) -> dict[str, Any]:
        provider = _required_text(pricing.provider, "provider")
        model = _required_text(pricing.model, "model")
        currency = _required_text(pricing.currency, "currency").upper()
        if len(currency) != 3:
            raise ValueError("currency must be a three-letter code such as EUR or USD.")
        input_price = _nonnegative_price(
            pricing.input_price_per_million,
            "input_price_per_million",
        )
        output_price = _nonnegative_price(
            pricing.output_price_per_million,
            "output_price_per_million",
        )
        updated_at = utc_now_iso()
        with self.connections.connect() as connection:
            connection.execute(
                """
                INSERT INTO model_pricing(
                    provider, model, currency, input_price_per_million,
                    output_price_per_million, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, model) DO UPDATE SET
                    currency=excluded.currency,
                    input_price_per_million=excluded.input_price_per_million,
                    output_price_per_million=excluded.output_price_per_million,
                    updated_at=excluded.updated_at
                """,
                (provider, model, currency, input_price, output_price, updated_at),
            )
            connection.commit()
        return {
            "provider": provider,
            "model": model,
            "currency": currency,
            "input_price_per_million": input_price,
            "output_price_per_million": output_price,
            "updated_at": updated_at,
        }


def _where_clause(filters: ModelCallFilter) -> tuple[str, tuple[object, ...]]:
    clauses: list[str] = []
    parameters: list[object] = []
    for column, value in (
        ("mc.recorded_at >=", filters.since),
        ("mc.provider =", filters.provider),
        ("mc.model =", filters.model),
        ("mc.operation =", filters.operation),
        ("mc.status =", filters.status),
    ):
        normalized = str(value or "").strip()
        if not normalized:
            continue
        clauses.append(f"{column} ?")
        parameters.append(normalized)
    return ("WHERE " + " AND ".join(clauses) if clauses else "", tuple(parameters))


def _row_to_call(row: Any) -> dict[str, Any]:
    input_tokens = row["input_tokens"]
    output_tokens = row["output_tokens"]
    input_price = row["input_price_per_million"]
    output_price = row["output_price_per_million"]
    estimated_cost: float | None = None
    if input_price is not None and output_price is not None:
        estimated_cost = (
            float(input_tokens or 0) * float(input_price)
            + float(output_tokens or 0) * float(output_price)
        ) / 1_000_000.0
    generation_duration_ms = row["generation_duration_ms"]
    generated_rate: float | None = None
    if output_tokens is not None and generation_duration_ms not in (None, 0):
        generated_rate = float(output_tokens) / (float(generation_duration_ms) / 1000.0)
    provider = str(row["provider"])
    model = str(row["model"] or "unknown")
    return {
        "call_id": row["call_id"],
        "recorded_at": row["recorded_at"],
        "started_at": row["started_at"],
        "trace_id": row["trace_id"],
        "job_id": row["job_id"],
        "receipt_id": row["receipt_id"],
        "query_id": row["query_id"],
        "operation": row["operation"],
        "provider": provider,
        "model": row["model"],
        "model_key": f"{provider}/{model}",
        "endpoint": row["endpoint"],
        "status": row["status"],
        "attempt": row["attempt"],
        "duration_ms": round(float(row["duration_ms"] or 0.0), 3),
        "model_total_duration_ms": row["model_total_duration_ms"],
        "model_load_duration_ms": row["model_load_duration_ms"],
        "prompt_evaluation_duration_ms": row["prompt_evaluation_duration_ms"],
        "generation_duration_ms": generation_duration_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(input_tokens or 0) + int(output_tokens or 0),
        "input_characters": row["input_characters"],
        "output_characters": row["output_characters"],
        "token_source": row["token_source"],
        "configured_context_window": row["configured_context_window"],
        "stop_reason": row["stop_reason"],
        "error": row["error"],
        "generated_tokens_per_second": (
            round(generated_rate, 2) if generated_rate is not None else None
        ),
        "estimated_cost": round(estimated_cost, 8) if estimated_cost is not None else None,
        "currency": row["currency"],
        "input_price_per_million": input_price,
        "output_price_per_million": output_price,
    }


def _group_records(records: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = str(record.get(key) or "unknown")
        grouped.setdefault(value, []).append(record)
    results: list[dict[str, Any]] = []
    for name, group in grouped.items():
        costs = [item["estimated_cost"] for item in group if item["estimated_cost"] is not None]
        results.append(
            {
                "name": name,
                "call_count": len(group),
                "input_tokens": sum(int(item["input_tokens"] or 0) for item in group),
                "output_tokens": sum(int(item["output_tokens"] or 0) for item in group),
                "average_duration_ms": _average(
                    [float(item["duration_ms"]) for item in group]
                ),
                "estimated_cost": (
                    round(sum(float(cost) for cost in costs), 8) if costs else None
                ),
                "priced_call_count": len(costs),
                "failed_call_count": sum(1 for item in group if item["status"] == "failed"),
            }
        )
    return sorted(results, key=lambda item: (-int(item["call_count"]), str(item["name"])))


def _single_currency(records: Sequence[dict[str, Any]]) -> str | None:
    currencies = {str(record["currency"]) for record in records if record.get("currency")}
    return next(iter(currencies)) if len(currencies) == 1 else None


def _average(values: Sequence[float]) -> float | None:
    return None if not values else round(sum(values) / len(values), 3)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    index = max(0, min(len(values) - 1, math.ceil(len(values) * percentile) - 1))
    return round(float(values[index]), 3)


def _required_text(value: str, name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _nonnegative_price(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return parsed


__all__ = ["SQLiteModelCallRepository"]
