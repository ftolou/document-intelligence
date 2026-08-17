"""Read and pricing contracts for model-call observability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelCallFilter:
    """Stable filters supported by the model-call dashboard."""

    since: str | None = None
    provider: str | None = None
    model: str | None = None
    operation: str | None = None
    status: str | None = None


@dataclass(frozen=True, slots=True)
class ModelPricingInput:
    provider: str
    model: str
    currency: str
    input_price_per_million: float
    output_price_per_million: float
    cached_input_price_per_million: float | None = None
    cache_write_input_price_per_million: float | None = None
    pricing_source: str | None = None
    effective_from: str | None = None


class ModelCallRepository(Protocol):
    """Query model calls and maintain the user-configured pricing catalog."""

    def summary(self, filters: ModelCallFilter) -> dict[str, Any]: ...

    def list_calls(
        self,
        filters: ModelCallFilter,
        *,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]: ...

    def list_pricing(self) -> list[dict[str, Any]]: ...

    def list_models(self) -> list[dict[str, Any]]: ...

    def upsert_pricing(self, pricing: ModelPricingInput) -> dict[str, Any]: ...


__all__ = ["ModelCallFilter", "ModelCallRepository", "ModelPricingInput"]
