"""Structured LLM SQL planning for the isolated RAG-SQL strategy."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from receipt_intelligence.application.generation import (
    LegacyGenerateFunction,
    invoke_generation,
)
from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    LlmGateway,
    ModelCallMetrics,
)
from receipt_intelligence.prompts import render_prompt_template
from receipt_intelligence.rag_sql.filter_definitions import (
    get_filter_definition,
    render_sql_filter_binding_catalog,
)
from receipt_intelligence.rag_sql.models import (
    JsonScalar,
    QuestionAnalysisResult,
    RagSqlPlanPayload,
    RagSqlPlanResult,
    ResolvedQueryFilter,
    ResolvedSemanticEntity,
)
from receipt_intelligence.rag_sql.schema_catalog import (
    DEFAULT_SCHEMA_CATALOG,
    StaticSchemaCatalog,
)
from receipt_intelligence.rag_sql.sql_dialect import get_sql_dialect_profile


class RagSqlPlanningError(RuntimeError):
    """Raised when the LLM cannot produce a valid structured SQL plan."""

    def __init__(
        self,
        message: str,
        *,
        ollama_calls: list[ModelCallMetrics] | None = None,
    ) -> None:
        super().__init__(message)
        self.ollama_calls = list(ollama_calls or [])


@dataclass(frozen=True)
class RagSqlPlannerConfig:
    enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    model: str = "gemma4:latest"
    num_ctx: int = 6144
    num_predict: int = 2048
    timeout_seconds: float = 120.0
    retry_count: int = 1
    format_json: bool = True
    keep_alive: str | None = None
    maximum_rows: int = 100
    sql_dialect: str = "sqlite"

    def __post_init__(self) -> None:
        if not str(self.ollama_url or "").strip():
            raise ValueError("ollama_url must not be empty.")
        if not str(self.model or "").strip():
            raise ValueError("model must not be empty.")
        if self.num_ctx <= 0 or self.num_predict <= 0:
            raise ValueError("num_ctx and num_predict must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.retry_count < 0:
            raise ValueError("retry_count must not be negative.")
        if self.maximum_rows <= 0 or self.maximum_rows > 1000:
            raise ValueError("maximum_rows must be between 1 and 1000.")
        get_sql_dialect_profile(self.sql_dialect)


class RagSqlPlanner:
    def __init__(
        self,
        config: RagSqlPlannerConfig,
        *,
        schema_catalog: StaticSchemaCatalog = DEFAULT_SCHEMA_CATALOG,
        llm_gateway: LlmGateway | None = None,
        generate: LegacyGenerateFunction | None = None,
    ) -> None:
        self.config = config
        self.schema_catalog = schema_catalog
        self.sql_dialect = get_sql_dialect_profile(config.sql_dialect)
        if self.schema_catalog.dialect_profile.name != self.sql_dialect.name:
            raise ValueError(
                "planner SQL dialect must match schema catalog SQL dialect: "
                f"planner={self.sql_dialect.name!r}, "
                f"catalog={self.schema_catalog.dialect_profile.name!r}."
            )
        self.llm_gateway = llm_gateway
        self.generate = generate

    def plan(
        self,
        question: str,
        *,
        analysis: QuestionAnalysisResult,
        resolved_entities: Sequence[ResolvedQueryFilter | ResolvedSemanticEntity],
        protected_parameters: Mapping[str, JsonScalar],
        previous_plan: RagSqlPlanResult | None = None,
        validation_error: str | None = None,
    ) -> RagSqlPlanResult:
        normalized_question = " ".join(str(question or "").split()).strip()
        if not normalized_question:
            raise ValueError("question must not be empty.")
        if not self.config.enabled:
            raise RagSqlPlanningError("RAG-SQL planning is disabled.")
        if (previous_plan is None) != (validation_error is None):
            raise ValueError(
                "previous_plan and validation_error must either both be provided or both be omitted."
            )

        resolved_payload = []
        for resolved_filter in resolved_entities:
            if isinstance(resolved_filter, ResolvedQueryFilter):
                parameter_names = [
                    name
                    for name in protected_parameters
                    if name.startswith(f"{resolved_filter.filter_id}_")
                ]
                resolved_payload.append(
                    {
                        "filter_id": resolved_filter.filter_id,
                        "field": resolved_filter.field,
                        "operator": resolved_filter.operator,
                        "original_value": resolved_filter.original_value,
                        "status": resolved_filter.status,
                        "resolved_values": resolved_filter.resolved_values,
                        "protected_parameters": {
                            name: protected_parameters[name] for name in sorted(parameter_names)
                        },
                    }
                )
            else:
                parameter_names = [
                    name
                    for name in protected_parameters
                    if name.startswith(f"{resolved_filter.entity_id}_item_")
                ]
                resolved_payload.append(
                    {
                        "filter_id": resolved_filter.entity_id,
                        "field": "product",
                        "operator": "matches",
                        "original_value": resolved_filter.search_text,
                        "status": resolved_filter.status,
                        "resolved_values": resolved_filter.selected_item_ids,
                        "protected_parameters": {
                            name: protected_parameters[name] for name in sorted(parameter_names)
                        },
                    }
                )

        started = time.perf_counter()
        previous_error: str | None = None
        last_error: Exception | None = None
        attempts = max(1, self.config.retry_count + 1)
        ollama_calls: list[ModelCallMetrics] = []
        previous_raw_response: str | None = None
        validation_repair_block = ""
        if previous_plan is not None and validation_error is not None:
            previous_payload = previous_plan.model_dump(
                mode="json",
                include=set(RagSqlPlanPayload.model_fields),
            )
            validation_repair_block = (
                "A previous complete SQL plan passed JSON/schema validation but failed "
                "deterministic SQL validation.\n\n"
                "Deterministic validation error:\n"
                f"{validation_error}\n\n"
                "Previous complete SQL plan:\n"
                f"{json.dumps(previous_payload, ensure_ascii=False, indent=2)}\n\n"
                "Return a complete replacement JSON plan. Correct the exact validation "
                "failure without changing the analytical meaning. Preserve every protected "
                "filter parameter name and value exactly. For grouped_rows, preserve all "
                "possible groups by using a deterministic ORDER BY and LIMIT "
                f"{self.config.maximum_rows} unless the original question explicitly requests "
                "fewer groups. Never use LIMIT 1 merely to satisfy validation. Do not explain "
                "the repair outside the JSON object."
            )

        for attempt in range(1, attempts + 1):
            response_schema = RagSqlPlanPayload.model_json_schema()
            retry_block = ""
            if previous_error:
                previous_response_block = (
                    f"\nPrevious invalid JSON response:\n{previous_raw_response}\n"
                    if previous_raw_response
                    else ""
                )
                retry_block = (
                    "The previous response failed the strict plan contract.\n"
                    "Exact validation error:\n"
                    f"{previous_error}\n"
                    f"{previous_response_block}"
                    "Return a complete corrected JSON plan. Keep result_entity as one "
                    "lowercase snake_case identifier. display_columns must contain only "
                    "SELECT output names or aliases, never SQL expressions such as COUNT(*). "
                    "Preserve all protected parameters exactly."
                )
            prompt = render_prompt_template(
                "rag_sql_planner.txt",
                SQL_DIALECT=self.sql_dialect.planner_label,
                TODAY=datetime.now(ZoneInfo("Europe/Berlin")).date().isoformat(),
                MAX_ROWS=self.config.maximum_rows,
                SCHEMA_CONTEXT=self.schema_catalog.render_for_prompt(),
                FILTER_BINDINGS=render_sql_filter_binding_catalog(),
                ANALYSIS_JSON=json.dumps(
                    analysis.model_dump(mode="json"), ensure_ascii=False, indent=2
                ),
                RESOLVED_FILTERS_JSON=json.dumps(resolved_payload, ensure_ascii=False, indent=2),
                QUESTION=normalized_question,
                VALIDATION_REPAIR_BLOCK=validation_repair_block,
                RETRY_BLOCK=retry_block,
            )
            try:
                generation = invoke_generation(
                    request=GenerationRequest(
                        model=self.config.model,
                        prompt=prompt,
                        operation="rag_sql_planning",
                        attempt=attempt,
                        num_ctx=self.config.num_ctx,
                        num_predict=self.config.num_predict,
                        temperature=0.0,
                        keep_alive=self.config.keep_alive,
                        timeout_seconds=self.config.timeout_seconds,
                        format_json=self.config.format_json,
                        response_json_schema=(response_schema if self.config.format_json else None),
                    ),
                    gateway=self.llm_gateway,
                    legacy_generate=self.generate,
                    legacy_base_url=self.config.ollama_url,
                )
                if generation.metrics is not None:
                    ollama_calls.append(generation.metrics)
                previous_raw_response = generation.text[:12000]
                payload = RagSqlPlanPayload.model_validate(
                    parse_json_from_llm(
                        generation,
                        response_json_schema=response_schema,
                    )
                )
                _validate_protected_parameters(payload, protected_parameters)
                return RagSqlPlanResult(
                    **payload.model_dump(mode="python"),
                    model=self.config.model,
                    attempts=attempt,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    ollama_calls=ollama_calls,
                )
            except Exception as exc:
                last_error = exc
                previous_error = f"{type(exc).__name__}: {exc}"

        duration_ms = (time.perf_counter() - started) * 1000.0
        raise RagSqlPlanningError(
            "RAG-SQL planning failed after "
            f"{attempts} attempt(s) in {duration_ms:.1f} ms: "
            f"{type(last_error).__name__ if last_error else 'UnknownError'}: {last_error}",
            ollama_calls=ollama_calls,
        ) from last_error

    def repair_after_validation_failure(
        self,
        question: str,
        *,
        analysis: QuestionAnalysisResult,
        resolved_entities: Sequence[ResolvedQueryFilter | ResolvedSemanticEntity],
        protected_parameters: Mapping[str, JsonScalar],
        previous_plan: RagSqlPlanResult,
        validation_error: str,
    ) -> RagSqlPlanResult:
        """Regenerate a complete plan from deterministic validator feedback."""

        validation_error_text = str(validation_error or "").strip()
        if not validation_error_text:
            raise ValueError("validation_error must not be empty.")
        if previous_plan.status != "ready":
            raise ValueError("Only a ready SQL plan can be repaired.")
        return self.plan(
            question,
            analysis=analysis,
            resolved_entities=resolved_entities,
            protected_parameters=protected_parameters,
            previous_plan=previous_plan,
            validation_error=validation_error_text,
        )


def build_protected_filter_parameters(
    filters: Sequence[ResolvedQueryFilter | ResolvedSemanticEntity],
) -> dict[str, JsonScalar]:
    """Create deterministic app-owned bindings for every resolved filter."""

    parameters: dict[str, JsonScalar] = {}
    for resolved_filter in filters:
        if isinstance(resolved_filter, ResolvedSemanticEntity):
            if resolved_filter.status != "resolved":
                continue
            for index, item_id in enumerate(resolved_filter.selected_item_ids):
                parameters[f"{resolved_filter.entity_id}_item_{index}"] = int(item_id)
            continue
        if resolved_filter.status != "resolved":
            continue
        suffix = get_filter_definition(resolved_filter.field).parameter_suffix
        for index, value in enumerate(resolved_filter.resolved_values):
            parameters[f"{resolved_filter.filter_id}_{suffix}_{index}"] = value
    return parameters


def build_protected_item_parameters(
    entities: Sequence[ResolvedSemanticEntity],
) -> dict[str, int]:
    """Compatibility wrapper for the former product-only contract."""

    return {
        name: int(value)
        for name, value in build_protected_filter_parameters(entities).items()
        if isinstance(value, int)
    }


def _validate_protected_parameters(
    payload: RagSqlPlanPayload,
    protected_parameters: Mapping[str, JsonScalar],
) -> None:
    if payload.status != "ready":
        return
    unexpected_protected = sorted(
        name
        for name in payload.parameters
        if re.fullmatch(r"[ef]\d{3}_[a-z][a-z0-9_]*_\d+", name) and name not in protected_parameters
    )
    if unexpected_protected:
        raise ValueError(
            f"The SQL plan introduced unknown protected filter parameters: {unexpected_protected}."
        )
    for name, expected_value in protected_parameters.items():
        if name not in payload.parameters:
            raise ValueError(f"The SQL plan omitted protected parameter {name!r}.")
        actual_value = payload.parameters[name]
        if isinstance(actual_value, bool) or actual_value != expected_value:
            raise ValueError(
                f"Protected parameter {name!r} changed from {expected_value!r} to {actual_value!r}."
            )


__all__ = [
    "RagSqlPlanner",
    "RagSqlPlannerConfig",
    "RagSqlPlanningError",
    "build_protected_filter_parameters",
    "build_protected_item_parameters",
]
