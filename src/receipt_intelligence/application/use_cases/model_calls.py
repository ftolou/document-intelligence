"""Application use cases for model-call usage and cost analytics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from receipt_intelligence.application.ports.model_calls import (
    ModelCallFilter,
    ModelCallRepository,
    ModelPricingInput,
)


class ModelCallUseCases:
    def __init__(self, repository: ModelCallRepository) -> None:
        self.repository = repository

    def summary(
        self,
        *,
        hours: int | None = 24,
        **filters: str | None,
    ) -> dict[str, Any]:
        return self.repository.summary(self._filters(hours=hours, **filters))

    def list_calls(
        self,
        *,
        hours: int | None = 24,
        limit: int = 100,
        offset: int = 0,
        **filters: str | None,
    ) -> list[dict[str, Any]]:
        return self.repository.list_calls(
            self._filters(hours=hours, **filters),
            limit=limit,
            offset=offset,
        )

    def pricing(self) -> list[dict[str, Any]]:
        return self.repository.list_pricing()

    def models(self) -> list[dict[str, Any]]:
        return self.repository.list_models()

    def save_pricing(self, payload: dict[str, Any]) -> dict[str, Any]:
        cached_price = payload.get("cached_input_price_per_million")
        cache_write_price = payload.get("cache_write_input_price_per_million")
        return self.repository.upsert_pricing(
            ModelPricingInput(
                provider=str(payload.get("provider") or ""),
                model=str(payload.get("model") or ""),
                currency=str(payload.get("currency") or "EUR"),
                input_price_per_million=float(payload.get("input_price_per_million") or 0),
                output_price_per_million=float(payload.get("output_price_per_million") or 0),
                cached_input_price_per_million=(
                    None if cached_price in (None, "") else float(cached_price)
                ),
                cache_write_input_price_per_million=(
                    None if cache_write_price in (None, "") else float(cache_write_price)
                ),
                pricing_source=str(payload.get("pricing_source") or "manual").strip() or "manual",
                effective_from=(
                    str(payload.get("effective_from") or "").strip() or None
                ),
            )
        )

    @staticmethod
    def _filters(
        *,
        hours: int | None,
        provider: str | None = None,
        model: str | None = None,
        operation: str | None = None,
        status: str | None = None,
    ) -> ModelCallFilter:
        since = None
        if hours is not None and hours > 0:
            since = (
                (datetime.now(UTC) - timedelta(hours=hours))
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
        return ModelCallFilter(
            since=since,
            provider=provider,
            model=model,
            operation=operation,
            status=status,
        )


__all__ = ["ModelCallUseCases"]
