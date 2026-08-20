"""Application runtime composition for text-embedding providers."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.adapters.embeddings import (
    EmbeddingProviderConfig,
    build_embedding_gateway,
)
from receipt_intelligence.application.ports.embeddings import CloseableEmbeddingGateway


class EmbeddingRuntimeSettings(Protocol):
    """Settings required to compose the active embedding provider."""

    RAG_EMBEDDING_PROVIDER: str
    RAG_EMBEDDING_MODEL: str
    RAG_EMBEDDING_BASE_URL: str | None
    RAG_EMBEDDING_DIMENSIONS: int | None
    RAG_EMBEDDING_TIMEOUT_SECONDS: float
    RAG_EMBEDDING_KEEP_ALIVE: str | None
    OPENAI_API_KEY: str


def build_embedding_provider_config_from_settings(
    settings_source: EmbeddingRuntimeSettings | None = None,
) -> EmbeddingProviderConfig:
    """Translate runtime settings into one provider configuration."""

    if settings_source is None:
        from receipt_intelligence import settings as runtime_settings

        settings_source = runtime_settings

    provider = str(settings_source.RAG_EMBEDDING_PROVIDER or "").strip().lower()
    return EmbeddingProviderConfig(
        provider=provider,
        model=settings_source.RAG_EMBEDDING_MODEL,
        base_url=settings_source.RAG_EMBEDDING_BASE_URL,
        api_key=settings_source.OPENAI_API_KEY if provider == "openai" else None,
        dimensions=settings_source.RAG_EMBEDDING_DIMENSIONS,
        timeout_seconds=settings_source.RAG_EMBEDDING_TIMEOUT_SECONDS,
        keep_alive=(settings_source.RAG_EMBEDDING_KEEP_ALIVE if provider == "ollama" else None),
    )


def build_embedding_gateway_from_settings(
    settings_source: EmbeddingRuntimeSettings | None = None,
) -> CloseableEmbeddingGateway:
    """Build the configured embedding adapter for the current runtime."""

    return build_embedding_gateway(build_embedding_provider_config_from_settings(settings_source))


__all__ = [
    "EmbeddingRuntimeSettings",
    "build_embedding_gateway_from_settings",
    "build_embedding_provider_config_from_settings",
]
