"""SQLite read model and event sink for model-call telemetry."""

from __future__ import annotations

import json
import math
import re
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
                        record["call_id"],
                        record["recorded_at"],
                        record["started_at"],
                        record.get("trace_id"),
                        record.get("job_id"),
                        record.get("receipt_id"),
                        record.get("query_id"),
                        record["operation"],
                        record["provider"],
                        record.get("model"),
                        record["endpoint"],
                        record["status"],
                        int(record.get("attempt") or 1),
                        float(record.get("duration_ms") or 0.0),
                        record.get("input_tokens"),
                        record.get("output_tokens"),
                        record.get("input_characters"),
                        record.get("output_characters"),
                        record.get("token_source") or "unavailable",
                        record.get("model_total_duration_ms"),
                        record.get("model_load_duration_ms"),
                        record.get("prompt_evaluation_duration_ms"),
                        record.get("generation_duration_ms"),
                        record.get("configured_context_window"),
                        record.get("stop_reason"),
                        record.get("error"),
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
                       p.cached_input_price_per_million,
                       p.cache_write_input_price_per_million,
                       p.output_price_per_million, p.pricing_source, p.effective_from
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
        priced_costs = [
            record["estimated_cost"] for record in records if record["estimated_cost"] is not None
        ]
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
                round(sum(float(value) for value in priced_costs), 8) if priced_costs else None
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
                       p.cached_input_price_per_million,
                       p.cache_write_input_price_per_million,
                       p.output_price_per_million, p.pricing_source, p.effective_from
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
                       cached_input_price_per_million,
                       cache_write_input_price_per_million,
                       output_price_per_million, pricing_source, effective_from, updated_at
                FROM model_pricing
                ORDER BY provider, model
                """
            ).fetchall()
        return [_pricing_row(dict(row)) for row in rows]

    def list_models(self) -> list[dict[str, Any]]:
        with self.connections.connect_read_only() as connection:
            observed_rows = connection.execute(
                """
                SELECT provider, model, COUNT(*) AS call_count, MAX(recorded_at) AS last_seen_at
                FROM model_calls
                WHERE model IS NOT NULL AND trim(model) <> ''
                GROUP BY provider, model
                ORDER BY MAX(recorded_at) DESC
                """
            ).fetchall()
            pricing_rows = connection.execute(
                """
                SELECT provider, model
                FROM model_pricing
                ORDER BY provider, model
                """
            ).fetchall()

        catalog: dict[tuple[str, str], dict[str, Any]] = {}
        for row in observed_rows:
            provider = str(row["provider"])
            model = str(row["model"])
            catalog[(provider, model)] = {
                **_model_identity(provider, model),
                "observed": True,
                "call_count": int(row["call_count"] or 0),
                "last_seen_at": row["last_seen_at"],
                "has_pricing": False,
            }
        for row in pricing_rows:
            provider = str(row["provider"])
            model = str(row["model"])
            entry = catalog.setdefault(
                (provider, model),
                {
                    **_model_identity(provider, model),
                    "observed": False,
                    "call_count": 0,
                    "last_seen_at": None,
                    "has_pricing": False,
                },
            )
            entry["has_pricing"] = True

        return sorted(
            catalog.values(),
            key=lambda item: (
                not bool(item["observed"]),
                str(item["provider_display_name"]),
                str(item["model_display_name"]),
            ),
        )

    def upsert_pricing(self, pricing: ModelPricingInput) -> dict[str, Any]:
        provider = _required_text(pricing.provider, "provider")
        model = _required_text(pricing.model, "model")
        provider, model = self._canonical_model_identity(provider, model)
        currency = _required_text(pricing.currency, "currency").upper()
        if len(currency) != 3:
            raise ValueError("currency must be a three-letter code such as EUR or USD.")
        input_price = _nonnegative_price(
            pricing.input_price_per_million,
            "input_price_per_million",
        )
        cached_input_price = _optional_nonnegative_price(
            pricing.cached_input_price_per_million,
            "cached_input_price_per_million",
        )
        cache_write_input_price = _optional_nonnegative_price(
            pricing.cache_write_input_price_per_million,
            "cache_write_input_price_per_million",
        )
        output_price = _nonnegative_price(
            pricing.output_price_per_million,
            "output_price_per_million",
        )
        pricing_source = str(pricing.pricing_source or "manual").strip() or "manual"
        effective_from = str(pricing.effective_from or "").strip() or None
        updated_at = utc_now_iso()
        with self.connections.connect() as connection:
            connection.execute(
                """
                INSERT INTO model_pricing(
                    provider, model, currency, input_price_per_million,
                    cached_input_price_per_million,
                    cache_write_input_price_per_million,
                    output_price_per_million, pricing_source, effective_from, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, model) DO UPDATE SET
                    currency=excluded.currency,
                    input_price_per_million=excluded.input_price_per_million,
                    cached_input_price_per_million=excluded.cached_input_price_per_million,
                    cache_write_input_price_per_million=excluded.cache_write_input_price_per_million,
                    output_price_per_million=excluded.output_price_per_million,
                    pricing_source=excluded.pricing_source,
                    effective_from=excluded.effective_from,
                    updated_at=excluded.updated_at
                """,
                (
                    provider,
                    model,
                    currency,
                    input_price,
                    cached_input_price,
                    cache_write_input_price,
                    output_price,
                    pricing_source,
                    effective_from,
                    updated_at,
                ),
            )
            connection.commit()
        return _pricing_row(
            {
                "provider": provider,
                "model": model,
                "currency": currency,
                "input_price_per_million": input_price,
                "cached_input_price_per_million": cached_input_price,
                "cache_write_input_price_per_million": cache_write_input_price,
                "output_price_per_million": output_price,
                "pricing_source": pricing_source,
                "effective_from": effective_from,
                "updated_at": updated_at,
            }
        )

    def _canonical_model_identity(self, provider: str, model: str) -> tuple[str, str]:
        target = _model_identity_key(provider, model)
        with self.connections.connect_read_only() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT provider, model
                FROM model_calls
                WHERE model IS NOT NULL AND trim(model) <> ''
                """
            ).fetchall()
        matches = [
            (str(row["provider"]), str(row["model"]))
            for row in rows
            if _model_identity_key(str(row["provider"]), str(row["model"])) == target
        ]
        if len(matches) == 1:
            return matches[0]
        return provider.strip().lower(), model.strip()


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
    input_tokens = _optional_int(row["input_tokens"])
    output_tokens = _optional_int(row["output_tokens"])
    attributes = _json_object(row["attributes_json"])
    cached_input_tokens = _bounded_component(
        _optional_int(attributes.get("cached_input_tokens")),
        input_tokens,
    )
    remaining_after_cached = (
        max(0, int(input_tokens or 0) - int(cached_input_tokens or 0))
        if input_tokens is not None
        else None
    )
    cache_write_input_tokens = _bounded_component(
        _optional_int(attributes.get("cache_write_input_tokens")),
        remaining_after_cached,
    )
    standard_input_tokens = (
        max(
            0,
            int(input_tokens or 0)
            - int(cached_input_tokens or 0)
            - int(cache_write_input_tokens or 0),
        )
        if input_tokens is not None
        else None
    )
    reasoning_output_tokens = _optional_int(attributes.get("reasoning_output_tokens"))

    input_price = row["input_price_per_million"]
    cached_input_price = row["cached_input_price_per_million"]
    cache_write_input_price = row["cache_write_input_price_per_million"]
    output_price = row["output_price_per_million"]
    estimated_cost, cost_breakdown, missing_price_components = _estimated_cost(
        standard_input_tokens=standard_input_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        output_tokens=output_tokens,
        input_price=input_price,
        cached_input_price=cached_input_price,
        cache_write_input_price=cache_write_input_price,
        output_price=output_price,
    )

    generation_duration_ms = row["generation_duration_ms"]
    generated_rate: float | None = None
    if output_tokens is not None and generation_duration_ms not in (None, 0):
        generated_rate = float(output_tokens) / (float(generation_duration_ms) / 1000.0)
    provider = str(row["provider"])
    model = str(row["model"] or "unknown")
    identity = _model_identity(provider, model)
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
        "provider_display_name": identity["provider_display_name"],
        "model_display_name": identity["model_display_name"],
        "display_name": identity["display_name"],
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
        "standard_input_tokens": standard_input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
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
        "estimated_cost_breakdown": cost_breakdown,
        "missing_price_components": missing_price_components,
        "currency": row["currency"],
        "input_price_per_million": input_price,
        "cached_input_price_per_million": cached_input_price,
        "cache_write_input_price_per_million": cache_write_input_price,
        "output_price_per_million": output_price,
        "pricing_source": row["pricing_source"],
        "pricing_effective_from": row["effective_from"],
    }


def _estimated_cost(
    *,
    standard_input_tokens: int | None,
    cached_input_tokens: int | None,
    cache_write_input_tokens: int | None,
    output_tokens: int | None,
    input_price: Any,
    cached_input_price: Any,
    cache_write_input_price: Any,
    output_price: Any,
) -> tuple[float | None, dict[str, float] | None, list[str]]:
    components = (
        ("standard_input", int(standard_input_tokens or 0), input_price),
        ("cached_input", int(cached_input_tokens or 0), cached_input_price),
        ("cache_write_input", int(cache_write_input_tokens or 0), cache_write_input_price),
        ("output", int(output_tokens or 0), output_price),
    )
    missing = [name for name, tokens, price in components if tokens > 0 and price is None]
    if missing:
        return None, None, missing
    if not any(tokens > 0 for _, tokens, _ in components):
        return 0.0, {name: 0.0 for name, _, _ in components}, []

    breakdown: dict[str, float] = {}
    total = 0.0
    for name, tokens, price in components:
        component_cost = (float(tokens) * float(price or 0.0)) / 1_000_000.0
        breakdown[name] = round(component_cost, 8)
        total += component_cost
    return total, breakdown, []


def _pricing_row(row: dict[str, Any]) -> dict[str, Any]:
    provider = str(row.get("provider") or "")
    model = str(row.get("model") or "")
    return {**row, **_model_identity(provider, model)}


def _model_identity(provider: str, model: str) -> dict[str, str]:
    provider_display = _provider_display_name(provider)
    model_display = _model_display_name(provider, model)
    return {
        "provider": provider,
        "model": model,
        "provider_display_name": provider_display,
        "model_display_name": model_display,
        "display_name": f"{provider_display} — {model_display}",
    }


def _provider_display_name(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    return {
        "openai": "OpenAI",
        "ollama": "Ollama",
    }.get(normalized, str(provider or "").strip() or "Unknown")


def _model_display_name(provider: str, model: str) -> str:
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip()
    if normalized_provider == "openai" and normalized_model.lower().startswith("gpt-"):
        suffix = normalized_model[4:]
        parts = suffix.split("-")
        if parts:
            version = parts[0]
            tier = " ".join(part.capitalize() for part in parts[1:])
            return f"GPT-{version}{f' {tier}' if tier else ''}"
    return normalized_model or "Unknown"


def compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _model_identity_key(provider: str, model: str) -> tuple[str, str]:
    # compact = lambda value: re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return compact(provider), compact(model)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        payload = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _bounded_component(value: int | None, maximum: int | None) -> int | None:
    if value is None:
        return 0 if maximum is not None else None
    if maximum is None:
        return value
    return max(0, min(int(value), int(maximum)))


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


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
                "average_duration_ms": _average([float(item["duration_ms"]) for item in group]),
                "estimated_cost": (round(sum(float(cost) for cost in costs), 8) if costs else None),
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


def _optional_nonnegative_price(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_price(value, name)


__all__ = ["SQLiteModelCallRepository"]
