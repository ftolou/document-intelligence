"""Typed query-filter resolution before SQL planning.

The analyzer selects from a small domain vocabulary. This registry dispatches
those filters to deterministic or semantic resolvers without exposing storage
columns or SQL generation to the LLM.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from typing import Protocol

from receipt_intelligence.application.ports.llm import ModelCallMetrics
from receipt_intelligence.rag.candidate_resolver import CandidateResolver
from receipt_intelligence.rag_sql.filter_definitions import (
    FILTER_DEFINITIONS,
    FilterField,
    FilterResolutionStrategy,
)
from receipt_intelligence.rag_sql.graph_support import SemanticRetriever
from receipt_intelligence.rag_sql.models import FilterScalar, QueryFilter, ResolvedQueryFilter


class FilterResolutionError(RuntimeError):
    """Raised when a filter resolver cannot complete reliably."""


class FilterValueCatalog(Protocol):
    """Read approved canonical values for low-cardinality filter fields."""

    def values(self, field: FilterField) -> Sequence[str]: ...


class QueryFilterResolver(Protocol):
    def resolve(
        self,
        query_filter: QueryFilter,
        *,
        user_question: str,
        language: str = "en",
    ) -> FilterResolutionBundle: ...


class EmptyFilterValueCatalog:
    def values(self, field: FilterField) -> Sequence[str]:
        del field
        return ()


@dataclass(frozen=True, slots=True)
class FilterResolutionBundle:
    resolution: ResolvedQueryFilter
    duration_ms: float
    details: dict[str, object] = field(default_factory=dict)
    model_calls: list[ModelCallMetrics] = field(default_factory=list)


class QueryFilterResolverRegistry:
    """Dispatch reusable query filters to field-specific resolution policies."""

    def __init__(
        self,
        *,
        retriever: SemanticRetriever,
        product_resolver: CandidateResolver,
        catalog: FilterValueCatalog | None = None,
        retrieval_limit: int = 12,
        retrieval_minimum_score: float | None = None,
        fuzzy_threshold: float = 0.84,
        fuzzy_margin: float = 0.08,
    ) -> None:
        if retrieval_limit <= 0 or retrieval_limit > 100:
            raise ValueError("retrieval_limit must be between 1 and 100.")
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be between 0 and 1.")
        if not 0.0 <= fuzzy_margin <= 1.0:
            raise ValueError("fuzzy_margin must be between 0 and 1.")
        self.retriever = retriever
        self.product_resolver = product_resolver
        self.catalog = catalog or EmptyFilterValueCatalog()
        self.retrieval_limit = retrieval_limit
        self.retrieval_minimum_score = retrieval_minimum_score
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_margin = fuzzy_margin
        strategy_handlers: dict[
            FilterResolutionStrategy,
            Callable[[QueryFilter, str, str], FilterResolutionBundle],
        ] = {
            "semantic_product": self._resolve_product,
            "catalog": self._resolve_catalog_filter,
            "scalar": self._resolve_scalar_filter,
        }
        self._handlers: dict[
            FilterField,
            Callable[[QueryFilter, str, str], FilterResolutionBundle],
        ] = {
            field: strategy_handlers[definition.resolution_strategy]
            for field, definition in FILTER_DEFINITIONS.items()
        }

    def resolve(
        self,
        query_filter: QueryFilter,
        *,
        user_question: str,
        language: str = "en",
    ) -> FilterResolutionBundle:
        handler = self._handlers.get(query_filter.field)
        if handler is None:  # defensive; FilterField already constrains this
            raise FilterResolutionError(f"No resolver is registered for {query_filter.field!r}.")
        return handler(query_filter, user_question, language)

    def _resolve_product(
        self,
        query_filter: QueryFilter,
        user_question: str,
        language: str,
    ) -> FilterResolutionBundle:
        del language
        started = time.perf_counter()
        terms = _as_values(query_filter.value)
        selected_ids: list[int] = []
        uncertain_ids: list[int] = []
        retrievals: list[dict[str, object]] = []
        model_calls: list[ModelCallMetrics] = []

        for term_value in terms:
            term = str(term_value).strip()
            search_started = time.perf_counter()
            search_result = self.retriever.search(
                term,
                limit=self.retrieval_limit,
                minimum_score=self.retrieval_minimum_score,
            )
            retrievals.append(
                {
                    "search_text": term,
                    "returned_candidates": len(search_result.matches),
                    "total_candidates": search_result.total_candidates,
                    "duration_ms": (time.perf_counter() - search_started) * 1000.0,
                    "model": search_result.model,
                }
            )
            model_calls.extend(search_result.ollama_calls)
            resolution = self.product_resolver.resolve(
                term,
                search_result.matches,
                user_question=user_question,
            )
            model_calls.extend(resolution.ollama_calls)
            selected_ids.extend(resolution.selected_item_ids)
            uncertain_ids.extend(resolution.uncertain_item_ids)
            if resolution.status == "needs_clarification":
                return FilterResolutionBundle(
                    resolution=ResolvedQueryFilter(
                        filter_id=query_filter.filter_id,
                        field=query_filter.field,
                        operator=query_filter.operator,
                        original_value=query_filter.value,
                        status="needs_clarification",
                        candidate_values=_unique(uncertain_ids),
                        clarification_question=resolution.clarification_question,
                    ),
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    details={"retrievals": retrievals},
                    model_calls=model_calls,
                )
            if resolution.status == "not_found":
                return FilterResolutionBundle(
                    resolution=ResolvedQueryFilter(
                        filter_id=query_filter.filter_id,
                        field=query_filter.field,
                        operator=query_filter.operator,
                        original_value=query_filter.value,
                        status="not_found",
                    ),
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    details={"retrievals": retrievals},
                    model_calls=model_calls,
                )

        return FilterResolutionBundle(
            resolution=ResolvedQueryFilter(
                filter_id=query_filter.filter_id,
                field=query_filter.field,
                operator=query_filter.operator,
                original_value=query_filter.value,
                status="resolved",
                resolved_values=_unique(selected_ids),
            ),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            details={"retrievals": retrievals},
            model_calls=model_calls,
        )

    def _resolve_catalog_filter(
        self,
        query_filter: QueryFilter,
        user_question: str,
        language: str,
    ) -> FilterResolutionBundle:
        del user_question
        started = time.perf_counter()
        catalog_values = [str(value) for value in self.catalog.values(query_filter.field) if value]
        if not catalog_values:
            return FilterResolutionBundle(
                resolution=ResolvedQueryFilter(
                    filter_id=query_filter.filter_id,
                    field=query_filter.field,
                    operator=query_filter.operator,
                    original_value=query_filter.value,
                    status="not_found",
                ),
                duration_ms=(time.perf_counter() - started) * 1000.0,
                details={"catalog_size": 0},
            )

        resolved_values: list[str] = []
        ambiguous_values: list[str] = []
        for requested in _as_values(query_filter.value):
            matches = self._match_catalog_value(
                query_filter.field,
                str(requested),
                catalog_values,
                operator=query_filter.operator,
            )
            if not matches:
                return FilterResolutionBundle(
                    resolution=ResolvedQueryFilter(
                        filter_id=query_filter.filter_id,
                        field=query_filter.field,
                        operator=query_filter.operator,
                        original_value=query_filter.value,
                        status="not_found",
                    ),
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    details={"catalog_size": len(catalog_values)},
                )
            if len(matches) > 1 and query_filter.operator != "contains":
                ambiguous_values.extend(matches)
            else:
                resolved_values.extend(matches)

        if ambiguous_values:
            options = _unique(ambiguous_values)[:10]
            return FilterResolutionBundle(
                resolution=ResolvedQueryFilter(
                    filter_id=query_filter.filter_id,
                    field=query_filter.field,
                    operator=query_filter.operator,
                    original_value=query_filter.value,
                    status="needs_clarification",
                    candidate_values=options,
                    clarification_question=_catalog_clarification_question(
                        language,
                        query_filter.field,
                        options,
                    ),
                ),
                duration_ms=(time.perf_counter() - started) * 1000.0,
                details={"catalog_size": len(catalog_values)},
            )

        return FilterResolutionBundle(
            resolution=ResolvedQueryFilter(
                filter_id=query_filter.filter_id,
                field=query_filter.field,
                operator=query_filter.operator,
                original_value=query_filter.value,
                status="resolved",
                resolved_values=_unique(resolved_values),
            ),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            details={"catalog_size": len(catalog_values)},
        )

    def _resolve_scalar_filter(
        self,
        query_filter: QueryFilter,
        user_question: str,
        language: str,
    ) -> FilterResolutionBundle:
        del user_question, language
        started = time.perf_counter()
        values: list[FilterScalar] = []
        for value in _as_values(query_filter.value):
            if query_filter.field == "purchase_date":
                normalized = date.fromisoformat(str(value)).isoformat()
                values.append(normalized)
            elif query_filter.field == "amount":
                values.append(float(value))
            elif query_filter.field == "receipt_id":
                values.append(int(value))
            else:  # pragma: no cover - registry limits callers
                values.append(value)
        return FilterResolutionBundle(
            resolution=ResolvedQueryFilter(
                filter_id=query_filter.filter_id,
                field=query_filter.field,
                operator=query_filter.operator,
                original_value=query_filter.value,
                status="resolved",
                resolved_values=values,
            ),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _match_catalog_value(
        self,
        field: FilterField,
        requested: str,
        catalog_values: Sequence[str],
        *,
        operator: str,
    ) -> list[str]:
        normalize = _normalizer(field)
        requested_key = normalize(requested)
        if not requested_key:
            return []
        keyed = [(value, normalize(value)) for value in catalog_values]

        exact = [value for value, key in keyed if key == requested_key]
        if exact or operator == "equals":
            return _unique(exact)

        if operator == "contains":
            return _unique([value for value, key in keyed if requested_key in key])

        contained = [
            value for value, key in keyed if requested_key in key or (key and key in requested_key)
        ]
        if contained:
            return _unique(contained)

        ranked = sorted(
            (
                (SequenceMatcher(None, requested_key, key).ratio(), value)
                for value, key in keyed
                if key
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] < self.fuzzy_threshold:
            return []
        best_score = ranked[0][0]
        close = [value for score, value in ranked if best_score - score < self.fuzzy_margin]
        if operator == "in":
            return _unique(close)
        return _unique(close[:2])


def _catalog_clarification_question(
    language: str,
    field: FilterField,
    options: Sequence[FilterScalar],
) -> str:
    rendered = ", ".join(str(value) for value in options)
    if language == "de":
        labels = {
            "merchant": "Händler",
            "category": "Kategorie",
            "payment_method": "Zahlungsart",
            "currency": "Währung",
        }
        return f"Welche {labels.get(field, 'Angabe')} meinst du: {rendered}?"
    return f"Which {field.replace('_', ' ')} did you mean: {rendered}?"


def _normalizer(field: FilterField) -> Callable[[object], str]:
    if field == "currency":
        return lambda value: str(value or "").strip().upper()
    return _normalize_text


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _as_values(value: FilterScalar | list[FilterScalar]) -> list[FilterScalar]:
    return list(value) if isinstance(value, list) else [value]


def _unique(values: Sequence[FilterScalar]) -> list[FilterScalar]:
    result: list[FilterScalar] = []
    seen: set[tuple[type[object], object]] = set()
    for value in values:
        key = (type(value), value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


__all__ = [
    "EmptyFilterValueCatalog",
    "FilterResolutionBundle",
    "FilterResolutionError",
    "FilterValueCatalog",
    "QueryFilterResolver",
    "QueryFilterResolverRegistry",
]
