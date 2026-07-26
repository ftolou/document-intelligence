from __future__ import annotations

from typing import Any

from receipt_intelligence.rag_sql.filter_resolution import QueryFilterResolverRegistry
from receipt_intelligence.rag_sql.models import QueryFilter


class FailingRetriever:
    def search(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("Product retrieval must not run for catalog or scalar filters.")


class FailingProductResolver:
    def resolve(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("Product candidate resolution must not run for non-product filters.")


class StaticCatalog:
    def __init__(self, values_by_field: dict[str, list[str]]) -> None:
        self.values_by_field = values_by_field
        self.calls: list[str] = []

    def values(self, field: str) -> list[str]:
        self.calls.append(field)
        return list(self.values_by_field.get(field, []))


def _registry(catalog: StaticCatalog) -> QueryFilterResolverRegistry:
    return QueryFilterResolverRegistry(
        retriever=FailingRetriever(),
        product_resolver=FailingProductResolver(),
        catalog=catalog,
    )


def test_merchant_filter_resolves_to_canonical_database_value() -> None:
    catalog = StaticCatalog({"merchant": ["ARAL Tankstelle", "REWE-MARKT"]})

    bundle = _registry(catalog).resolve(
        QueryFilter(
            filter_id="f001",
            field="merchant",
            operator="matches",
            value="aral",
        ),
        user_question="What did I buy at ARAL?",
    )

    assert bundle.resolution.status == "resolved"
    assert bundle.resolution.resolved_values == ["ARAL Tankstelle"]
    assert catalog.calls == ["merchant"]
    assert bundle.model_calls == []


def test_merchant_filter_requests_clarification_for_multiple_canonical_matches() -> None:
    catalog = StaticCatalog(
        {
            "merchant": [
                "ARAL Tankstelle Nord",
                "ARAL Tankstelle Süd",
                "REWE-MARKT",
            ]
        }
    )

    bundle = _registry(catalog).resolve(
        QueryFilter(
            filter_id="f001",
            field="merchant",
            operator="matches",
            value="ARAL",
        ),
        user_question="Show my ARAL receipts.",
    )

    assert bundle.resolution.status == "needs_clarification"
    assert bundle.resolution.candidate_values == [
        "ARAL Tankstelle Nord",
        "ARAL Tankstelle Süd",
    ]
    assert "Which merchant" in str(bundle.resolution.clarification_question)


def test_scalar_filters_are_validated_without_retrieval() -> None:
    bundle = _registry(StaticCatalog({})).resolve(
        QueryFilter(
            filter_id="f001",
            field="purchase_date",
            operator="between",
            value=["2026-01-01", "2026-12-31"],
        ),
        user_question="What did I buy in 2026?",
    )

    assert bundle.resolution.status == "resolved"
    assert bundle.resolution.resolved_values == ["2026-01-01", "2026-12-31"]
