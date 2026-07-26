"""Single source of truth for supported receipt query-filter capabilities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

FilterField = Literal[
    "product",
    "merchant",
    "category",
    "purchase_date",
    "amount",
    "payment_method",
    "currency",
    "receipt_id",
]
FilterOperator = Literal[
    "matches",
    "equals",
    "contains",
    "in",
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "before",
    "after",
    "between",
]
FilterValueKind = Literal["text", "date", "number", "positive_integer"]
FilterResolutionStrategy = Literal["semantic_product", "catalog", "scalar"]


@dataclass(frozen=True, slots=True)
class FilterDefinition:
    field: FilterField
    allowed_operators: frozenset[FilterOperator]
    value_kind: FilterValueKind
    resolution_strategy: FilterResolutionStrategy
    parameter_suffix: str
    sql_columns: tuple[str, ...]


def _definition(
    field: FilterField,
    *,
    allowed_operators: set[FilterOperator],
    value_kind: FilterValueKind,
    resolution_strategy: FilterResolutionStrategy,
    parameter_suffix: str,
    sql_columns: tuple[str, ...],
) -> FilterDefinition:
    return FilterDefinition(
        field=field,
        allowed_operators=frozenset(allowed_operators),
        value_kind=value_kind,
        resolution_strategy=resolution_strategy,
        parameter_suffix=parameter_suffix,
        sql_columns=sql_columns,
    )


FILTER_DEFINITIONS: Mapping[FilterField, FilterDefinition] = MappingProxyType(
    {
        "product": _definition(
            "product",
            allowed_operators={"matches", "equals"},
            value_kind="text",
            resolution_strategy="semantic_product",
            parameter_suffix="item",
            sql_columns=("item_id",),
        ),
        "merchant": _definition(
            "merchant",
            allowed_operators={"matches", "equals", "contains", "in"},
            value_kind="text",
            resolution_strategy="catalog",
            parameter_suffix="merchant",
            sql_columns=("merchant",),
        ),
        "category": _definition(
            "category",
            allowed_operators={"matches", "equals", "contains", "in"},
            value_kind="text",
            resolution_strategy="catalog",
            parameter_suffix="category",
            sql_columns=("category",),
        ),
        "purchase_date": _definition(
            "purchase_date",
            allowed_operators={"equals", "before", "after", "between"},
            value_kind="date",
            resolution_strategy="scalar",
            parameter_suffix="date",
            sql_columns=("receipt_date",),
        ),
        "amount": _definition(
            "amount",
            allowed_operators={
                "equals",
                "greater_than",
                "greater_than_or_equal",
                "less_than",
                "less_than_or_equal",
                "between",
            },
            value_kind="number",
            resolution_strategy="scalar",
            parameter_suffix="amount",
            sql_columns=("grand_total", "line_total"),
        ),
        "payment_method": _definition(
            "payment_method",
            allowed_operators={"matches", "equals", "contains", "in"},
            value_kind="text",
            resolution_strategy="catalog",
            parameter_suffix="payment_method",
            sql_columns=("payment_method",),
        ),
        "currency": _definition(
            "currency",
            allowed_operators={"matches", "equals", "contains", "in"},
            value_kind="text",
            resolution_strategy="catalog",
            parameter_suffix="currency",
            sql_columns=("currency",),
        ),
        "receipt_id": _definition(
            "receipt_id",
            allowed_operators={"equals", "in"},
            value_kind="positive_integer",
            resolution_strategy="scalar",
            parameter_suffix="receipt_id",
            sql_columns=("receipt_id",),
        ),
    }
)


def render_analysis_filter_catalog() -> str:
    payload = {
        field: {
            "allowed_operators": sorted(definition.allowed_operators),
            "value_kind": definition.value_kind,
        }
        for field, definition in FILTER_DEFINITIONS.items()
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def render_sql_filter_binding_catalog() -> str:
    payload = {
        field: {
            "parameter_suffix": definition.parameter_suffix,
            "sql_columns": list(definition.sql_columns),
        }
        for field, definition in FILTER_DEFINITIONS.items()
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def get_filter_definition(field: FilterField) -> FilterDefinition:
    return FILTER_DEFINITIONS[field]


__all__ = [
    "FILTER_DEFINITIONS",
    "FilterDefinition",
    "FilterField",
    "FilterOperator",
    "FilterResolutionStrategy",
    "FilterValueKind",
    "get_filter_definition",
    "render_analysis_filter_catalog",
    "render_sql_filter_binding_catalog",
]
