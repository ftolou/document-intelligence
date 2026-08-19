"""Provider-neutral text-embedding contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from receipt_intelligence.application.ports.llm import ModelCallMetrics


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider returns an unusable response."""


class EmbeddingBatchResult(BaseModel):
    """Validated embedding batch returned by any provider.

    Provider adapters may expose provider-native timing counters for compatibility,
    while portable request diagnostics are carried in ``model_calls``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model: str = Field(min_length=1, max_length=200)
    vectors: list[list[float]] = Field(default_factory=list)
    dimension: int = Field(ge=0)
    total_duration_ns: int | None = Field(default=None, ge=0)
    load_duration_ns: int | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    prompt_eval_duration_ns: int | None = Field(default=None, ge=0)
    model_calls: list[ModelCallMetrics] = Field(
        default_factory=list,
        max_length=20,
        validation_alias=AliasChoices("model_calls", "ollama_calls"),
    )

    @model_validator(mode="after")
    def validate_vectors(self) -> Self:
        if not self.vectors:
            if self.dimension != 0:
                raise ValueError("An empty embedding batch must have dimension=0.")
            return self

        if self.dimension <= 0:
            raise ValueError("A non-empty embedding batch requires a positive dimension.")

        for index, vector in enumerate(self.vectors):
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Embedding {index} has dimension {len(vector)}, expected {self.dimension}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise ValueError(f"Embedding {index} contains a non-finite value.")

        return self

    @property
    def count(self) -> int:
        return len(self.vectors)

    @property
    def ollama_calls(self) -> list[ModelCallMetrics]:
        """Compatibility alias for callers using the pre-gateway field name."""

        return self.model_calls

    @classmethod
    def empty(cls, *, model: str) -> EmbeddingBatchResult:
        return cls(model=model, vectors=[], dimension=0)


class EmbeddingGateway(Protocol):
    """Provider-neutral contract used by indexing and semantic retrieval."""

    model: str

    def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult: ...


class CloseableEmbeddingGateway(EmbeddingGateway, Protocol):
    """Embedding gateway whose provider resources are owned by the caller."""

    def close(self) -> None: ...


__all__ = [
    "CloseableEmbeddingGateway",
    "EmbeddingBatchResult",
    "EmbeddingGateway",
    "EmbeddingProviderError",
]
