"""Typed contracts for the RAG-assisted SQL query engine."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from receipt_intelligence.application.ports.llm import ModelCallMetrics
from receipt_intelligence.rag_sql.filter_definitions import (
    FilterField,
    FilterOperator,
    get_filter_definition,
)

RAG_SQL_ANALYSIS_SCHEMA_VERSION = "rag_sql_question_analysis_v3"
RAG_SQL_LEGACY_ANALYSIS_SCHEMA_VERSION = "rag_sql_question_analysis_v2"
RAG_SQL_PLAN_SCHEMA_VERSION = "rag_sql_plan_v2"
RAG_SQL_ENGINE_VERSION = "rag_sql_engine_v2"

JsonScalar = str | int | float | None
FilterScalar = str | int | float
FilterValue = FilterScalar | list[FilterScalar]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QueryFilter(StrictModel):
    """A reusable typed constraint extracted from natural language."""

    filter_id: str = Field(pattern=r"^[ef]\d{3}$")
    field: FilterField
    operator: FilterOperator
    value: FilterValue

    @field_validator("value", mode="before")
    @classmethod
    def reject_boolean_values(cls, value: object) -> object:
        values = value if isinstance(value, list) else [value]
        if any(isinstance(item, bool) for item in values):
            raise ValueError("Boolean filter values are not allowed.")
        return value

    @model_validator(mode="after")
    def validate_field_contract(self) -> Self:
        scalar = not isinstance(self.value, list)
        values = self.value if isinstance(self.value, list) else [self.value]

        definition = get_filter_definition(self.field)
        if self.operator not in definition.allowed_operators:
            raise ValueError(f"Operator {self.operator!r} is not valid for {self.field}.")

        if definition.value_kind == "text":
            if self.operator == "in" and scalar:
                raise ValueError("The 'in' operator requires a list value.")
            if self.operator != "in" and not scalar:
                raise ValueError(f"Operator {self.operator!r} requires one scalar value.")
            if not all(isinstance(value, str) and value.strip() for value in values):
                raise ValueError(f"Filter {self.field!r} requires non-empty string values.")
        elif definition.value_kind == "date":
            if self.operator == "between":
                if scalar or len(values) != 2:
                    raise ValueError("purchase_date between requires exactly two values.")
            elif not scalar:
                raise ValueError(f"Operator {self.operator!r} requires one date value.")
            if not all(isinstance(value, str) and value.strip() for value in values):
                raise ValueError("purchase_date requires ISO date strings.")
            parsed_dates = [date.fromisoformat(str(value)) for value in values]
            if self.operator == "between" and parsed_dates[0] > parsed_dates[1]:
                raise ValueError("purchase_date between requires ascending date values.")
        elif definition.value_kind == "number":
            if self.operator == "between":
                if scalar or len(values) != 2:
                    raise ValueError("amount between requires exactly two numeric values.")
            elif not scalar:
                raise ValueError(f"Operator {self.operator!r} requires one numeric value.")
            if not all(isinstance(value, int | float) for value in values):
                raise ValueError("amount requires numeric values.")
            if self.operator == "between" and float(values[0]) > float(values[1]):
                raise ValueError("amount between requires ascending numeric values.")
        elif definition.value_kind == "positive_integer":
            if self.operator == "in" and scalar:
                raise ValueError("receipt_id in requires a list value.")
            if self.operator == "equals" and not scalar:
                raise ValueError("receipt_id equals requires one value.")
            if not all(isinstance(value, int) and value > 0 for value in values):
                raise ValueError("receipt_id requires positive integers.")
        return self


class SemanticEntity(StrictModel):
    """Legacy v2 product-only analysis contract retained for migration."""

    entity_id: str = Field(pattern=r"^e\d{3}$")
    search_text: str = Field(min_length=1, max_length=500)
    role: Literal["product_filter"] = "product_filter"


class QuestionAnalysisPayload(StrictModel):
    """Raw structured question-analysis response expected from the LLM."""

    schema_version: Literal[
        "rag_sql_question_analysis_v3",
        "rag_sql_question_analysis_v2",
    ] = RAG_SQL_ANALYSIS_SCHEMA_VERSION
    status: Literal["ready", "needs_clarification", "unsupported"]
    language: Literal["de", "en"] = "de"
    user_goal: str | None = Field(default=None, min_length=1, max_length=1500)
    target_entity: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    requested_operation: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    filters: list[QueryFilter] = Field(default_factory=list, max_length=8)
    clarification_question: str | None = Field(default=None, max_length=1000)
    reason: str | None = Field(default=None, max_length=1000)

    # Migration-only fields. They are accepted from v2 payloads and constructors,
    # but omitted from canonical model dumps and planner prompts.
    requires_product_resolution: bool = Field(default=False, exclude=True)
    entities: list[SemanticEntity] = Field(default_factory=list, max_length=4, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_entities(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy_entities = data.get("entities") or []
        if legacy_entities and "schema_version" not in data:
            data["schema_version"] = RAG_SQL_LEGACY_ANALYSIS_SCHEMA_VERSION

        def legacy_value(entity: object, field: str) -> object | None:
            if isinstance(entity, dict):
                return entity.get(field)
            return getattr(entity, field, None)

        if legacy_entities:
            legacy_ids = [legacy_value(entity, "entity_id") for entity in legacy_entities]
            expected_legacy_ids = [f"e{index:03d}" for index in range(1, len(legacy_ids) + 1)]
            if legacy_ids != expected_legacy_ids:
                raise ValueError(f"Semantic entity IDs must be sequential: {expected_legacy_ids}.")
        if not data.get("filters") and legacy_entities:
            data["filters"] = [
                {
                    "filter_id": str(legacy_value(entity, "entity_id") or f"e{index:03d}"),
                    "field": "product",
                    "operator": "matches",
                    "value": legacy_value(entity, "search_text"),
                }
                for index, entity in enumerate(legacy_entities, start=1)
            ]
        return data

    @model_validator(mode="after")
    def validate_analysis(self) -> Self:
        filter_ids = [query_filter.filter_id for query_filter in self.filters]
        if len(filter_ids) != len(set(filter_ids)):
            raise ValueError("Query filter IDs must be unique.")
        prefix = "e" if self.schema_version == RAG_SQL_LEGACY_ANALYSIS_SCHEMA_VERSION else "f"
        expected_ids = [f"{prefix}{index:03d}" for index in range(1, len(self.filters) + 1)]
        if filter_ids != expected_ids:
            raise ValueError(f"Query filter IDs must be sequential: {expected_ids}.")

        product_filters = [value for value in self.filters if value.field == "product"]
        self.requires_product_resolution = bool(product_filters)
        self.entities = [
            SemanticEntity(
                entity_id=f"e{index:03d}",
                search_text=str(query_filter.value),
            )
            for index, query_filter in enumerate(product_filters, start=1)
        ]

        if self.status == "ready":
            if not self.user_goal or not self.target_entity or not self.requested_operation:
                raise ValueError(
                    "ready status requires user_goal, target_entity, and requested_operation."
                )
            if self.clarification_question or self.reason:
                raise ValueError("ready status cannot include clarification_question or reason.")
        elif self.status == "needs_clarification":
            if not self.clarification_question:
                raise ValueError("needs_clarification status requires clarification_question.")
        elif self.status == "unsupported":
            if not self.reason:
                raise ValueError("unsupported status requires reason.")
            if self.clarification_question:
                raise ValueError("unsupported status cannot include clarification_question.")
        return self


class QuestionAnalysisResult(QuestionAnalysisPayload):
    model: str | None = Field(default=None, max_length=200)
    attempts: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    ollama_calls: list[ModelCallMetrics] = Field(default_factory=list, max_length=20)


class ResolvedQueryFilter(StrictModel):
    filter_id: str = Field(pattern=r"^[ef]\d{3}$")
    field: FilterField
    operator: FilterOperator
    original_value: FilterValue
    status: Literal["resolved", "needs_clarification", "not_found"]
    resolved_values: list[FilterScalar] = Field(default_factory=list, max_length=100)
    candidate_values: list[FilterScalar] = Field(default_factory=list, max_length=20)
    clarification_question: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_resolution_state(self) -> Self:
        if self.status == "resolved":
            if not self.resolved_values:
                raise ValueError("resolved filter requires resolved_values.")
            if self.candidate_values or self.clarification_question:
                raise ValueError(
                    "resolved filter cannot contain candidate_values or clarification_question."
                )
        elif self.status == "needs_clarification":
            if not self.candidate_values or not self.clarification_question:
                raise ValueError(
                    "needs_clarification requires candidate_values and clarification_question."
                )
            if self.resolved_values:
                raise ValueError("needs_clarification cannot contain resolved_values.")
        elif self.status == "not_found":
            if self.resolved_values or self.candidate_values or self.clarification_question:
                raise ValueError(
                    "not_found filter cannot contain resolved values, candidate values, "
                    "or clarification_question."
                )
        if self.field == "product":
            values = [*self.resolved_values, *self.candidate_values]
            if any(not isinstance(value, int) or value <= 0 for value in values):
                raise ValueError("Resolved product filters require positive item IDs.")
        return self


class ResolvedSemanticEntity(StrictModel):
    """Legacy product-only resolution model retained for compatibility."""

    entity_id: str = Field(pattern=r"^e\d{3}$")
    search_text: str = Field(min_length=1, max_length=500)
    status: Literal["resolved", "needs_clarification", "not_found"]
    selected_item_ids: list[int] = Field(default_factory=list, max_length=100)
    uncertain_item_ids: list[int] = Field(default_factory=list, max_length=100)
    clarification_question: str | None = Field(default=None, max_length=1000)

    @field_validator("selected_item_ids", "uncertain_item_ids")
    @classmethod
    def validate_positive_unique_ids(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("Item IDs must be positive integers.")
        if len(values) != len(set(values)):
            raise ValueError("Item IDs must not contain duplicates.")
        return values

    @model_validator(mode="after")
    def validate_resolution_state(self) -> Self:
        if set(self.selected_item_ids) & set(self.uncertain_item_ids):
            raise ValueError("Selected and uncertain item IDs must not overlap.")
        if self.status == "resolved":
            if not self.selected_item_ids:
                raise ValueError("resolved entity requires selected_item_ids.")
            if self.uncertain_item_ids or self.clarification_question:
                raise ValueError(
                    "resolved entity cannot contain uncertain IDs or clarification_question."
                )
        elif self.status == "needs_clarification":
            if not self.uncertain_item_ids or not self.clarification_question:
                raise ValueError(
                    "needs_clarification requires uncertain IDs and clarification_question."
                )
        elif self.status == "not_found":
            if self.selected_item_ids or self.uncertain_item_ids:
                raise ValueError("not_found entity cannot contain item IDs.")
        return self


class RagSqlPlanPayload(StrictModel):
    """Structured SQL plan returned by the LLM before deterministic validation."""

    schema_version: Literal["rag_sql_plan_v2"] = RAG_SQL_PLAN_SCHEMA_VERSION
    status: Literal["ready", "needs_clarification", "unsupported"]
    sql: str | None = Field(default=None, max_length=20000)
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)
    result_shape: Literal["scalar", "row", "rows", "grouped_rows"] | None = None
    result_entity: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    display_columns: list[str] = Field(default_factory=list, max_length=20)
    answer_instruction: str | None = Field(default=None, max_length=1000)
    clarification_question: str | None = Field(default=None, max_length=1000)
    reason: str | None = Field(default=None, max_length=1000)

    @field_validator("parameters", mode="before")
    @classmethod
    def reject_boolean_parameters_before_coercion(cls, values: object) -> object:
        if isinstance(values, dict):
            for name, value in values.items():
                if isinstance(value, bool):
                    raise ValueError(f"Boolean SQL parameter {name!r} is not allowed.")
        return values

    @field_validator("display_columns")
    @classmethod
    def validate_display_columns(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("display_columns must not contain duplicates.")
        for value in values:
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
                raise ValueError(f"Invalid display column name: {value!r}.")
        return values

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, values: dict[str, JsonScalar]) -> dict[str, JsonScalar]:
        if len(values) > 100:
            raise ValueError("At most 100 SQL parameters are allowed.")
        for name, value in values.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
                raise ValueError(f"Invalid SQL parameter name: {name!r}.")
            if isinstance(value, bool):
                raise ValueError(f"Boolean SQL parameter {name!r} is not allowed.")
            if isinstance(value, str):
                if len(value) > 500:
                    raise ValueError(f"SQL parameter {name!r} exceeds 500 characters.")
                if "\x00" in value:
                    raise ValueError(f"SQL parameter {name!r} contains a NUL byte.")
        return values

    @model_validator(mode="after")
    def validate_plan_status(self) -> Self:
        if self.status == "ready":
            if (
                not self.sql
                or not self.result_shape
                or not self.result_entity
                or not self.answer_instruction
            ):
                raise ValueError(
                    "ready status requires sql, result_shape, result_entity, "
                    "and answer_instruction."
                )
            if self.result_shape in {"row", "rows", "grouped_rows"} and not self.display_columns:
                raise ValueError(f"{self.result_shape} status requires display_columns.")
            if self.clarification_question or self.reason:
                raise ValueError("ready status cannot include clarification_question or reason.")
        elif self.status == "needs_clarification":
            if not self.clarification_question:
                raise ValueError("needs_clarification status requires clarification_question.")
            if (
                self.sql
                or self.parameters
                or self.result_shape
                or self.result_entity
                or self.display_columns
            ):
                raise ValueError(
                    "needs_clarification status cannot include executable SQL metadata."
                )
        elif self.status == "unsupported":
            if not self.reason:
                raise ValueError("unsupported status requires reason.")
            if (
                self.sql
                or self.parameters
                or self.result_shape
                or self.result_entity
                or self.display_columns
            ):
                raise ValueError("unsupported status cannot include executable SQL metadata.")
        return self


class RagSqlPlanResult(RagSqlPlanPayload):
    model: str | None = Field(default=None, max_length=200)
    attempts: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    ollama_calls: list[ModelCallMetrics] = Field(default_factory=list, max_length=20)


class ValidatedSqlPlan(StrictModel):
    sql: str = Field(min_length=1, max_length=20000)
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)
    result_shape: Literal["scalar", "row", "rows", "grouped_rows"]
    result_entity: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    display_columns: list[str] = Field(default_factory=list, max_length=20)
    answer_instruction: str = Field(min_length=1, max_length=1000)
    referenced_objects: list[str] = Field(default_factory=list)
    referenced_functions: list[str] = Field(default_factory=list)
    placeholder_names: list[str] = Field(default_factory=list)


class SqlExecutionResult(StrictModel):
    columns: list[str] = Field(default_factory=list, max_length=200)
    rows: list[dict[str, Any]] = Field(default_factory=list, max_length=1000)
    row_count: int = Field(ge=0)
    truncated: bool = False
    duration_ms: float = Field(default=0.0, ge=0.0)


class RagSqlResponse(StrictModel):
    strategy: Literal["rag_sql"] = "rag_sql"
    engine_version: Literal["rag_sql_engine_v2"] = RAG_SQL_ENGINE_VERSION
    question: str = Field(min_length=1, max_length=4000)
    status: Literal[
        "completed",
        "needs_clarification",
        "not_found",
        "insufficient_info",
        "unsupported",
        "error",
    ]
    answer: str = Field(default="", max_length=4000)
    data: SqlExecutionResult | None = None
    clarification_question: str | None = Field(default=None, max_length=1000)
    error_code: str | None = Field(default=None, max_length=100)
    error: str | None = Field(default=None, max_length=2000)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
