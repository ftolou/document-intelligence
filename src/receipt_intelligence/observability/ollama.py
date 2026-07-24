"""Compatibility exports for historical Ollama telemetry imports.

New feature code should depend on ``application.ports.llm.ModelCallMetrics`` and
``GenerationResult``. This module remains only to avoid breaking older callers.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from receipt_intelligence.adapters.llm.ollama_gateway import (
    model_metrics_from_ollama_payload,
)
from receipt_intelligence.application.ports.llm import (
    GenerationResult,
    ModelCallMetrics,
    metrics_to_diagnostics as metrics_to_diagnostics,
)


class OllamaCallMetrics(ModelCallMetrics):
    """Deprecated provider-specific metrics type kept for API compatibility."""

    provider: Literal["ollama"] = Field(default="ollama")

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        endpoint: Literal["generate", "embed"],
        model: str,
        request_duration_ms: float,
        input_count: int | None = None,
    ) -> OllamaCallMetrics:
        metrics = model_metrics_from_ollama_payload(
            payload,
            endpoint=endpoint,
            model=model,
            request_duration_ms=request_duration_ms,
            input_count=input_count,
        )
        return cls.model_validate(metrics.model_dump())


def metrics_from_payload(
    payload: dict[str, Any],
    *,
    endpoint: Literal["generate", "embed"],
    model: str,
    request_duration_ms: float,
    input_count: int | None = None,
) -> ModelCallMetrics:
    return model_metrics_from_ollama_payload(
        payload,
        endpoint=endpoint,
        model=model,
        request_duration_ms=request_duration_ms,
        input_count=input_count,
    )


def get_ollama_metrics(value: object) -> ModelCallMetrics | None:
    """Return explicit Ollama metrics from a generation result, if present."""

    if not isinstance(value, GenerationResult):
        return None
    metrics = value.metrics
    if metrics is None or metrics.provider != "ollama":
        return None
    return metrics


__all__ = [
    "GenerationResult",
    "ModelCallMetrics",
    "OllamaCallMetrics",
    "get_ollama_metrics",
    "metrics_from_payload",
    "metrics_to_diagnostics",
]
