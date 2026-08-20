"""Composition helpers for text-embedding provider adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

from receipt_intelligence.application.ports.embeddings import CloseableEmbeddingGateway

from .ollama_gateway import OllamaEmbeddingGateway
from .openai_gateway import OpenAIEmbeddingGateway


@dataclass(frozen=True, slots=True)
class EmbeddingProviderConfig:
    """Provider connection settings kept outside semantic retrieval logic."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    dimensions: int | None = None
    timeout_seconds: float = 120.0
    keep_alive: str | None = None

    def __post_init__(self) -> None:
        provider = str(self.provider or "").strip().lower()
        model = str(self.model or "").strip()
        base_url = str(self.base_url or "").strip().rstrip("/") or None
        api_key = str(self.api_key or "").strip() or None
        keep_alive = str(self.keep_alive or "").strip() or None

        if not provider:
            raise ValueError("provider must not be empty.")
        if not model:
            raise ValueError("model must not be empty.")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be positive when provided.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "keep_alive", keep_alive)


def build_embedding_gateway(config: EmbeddingProviderConfig) -> CloseableEmbeddingGateway:
    """Create one embedding adapter from an explicit provider configuration."""

    if config.provider == "ollama":
        if not config.base_url:
            raise ValueError("Ollama embeddings require base_url.")
        if config.dimensions is not None:
            raise ValueError(
                "Explicit embedding dimensions are not supported by the Ollama adapter."
            )
        return OllamaEmbeddingGateway(
            base_url=config.base_url,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            keep_alive=config.keep_alive,
        )

    if config.provider == "openai":
        if not config.api_key:
            raise ValueError("OpenAI embeddings require an API key.")
        if config.base_url:
            return OpenAIEmbeddingGateway(
                api_key=config.api_key,
                model=config.model,
                base_url=config.base_url,
                dimensions=config.dimensions,
                timeout_seconds=config.timeout_seconds,
            )
        return OpenAIEmbeddingGateway(
            api_key=config.api_key,
            model=config.model,
            dimensions=config.dimensions,
            timeout_seconds=config.timeout_seconds,
        )

    raise ValueError(
        f"Unsupported embedding provider {config.provider!r}. Supported providers: ollama, openai."
    )


__all__ = ["EmbeddingProviderConfig", "build_embedding_gateway"]
