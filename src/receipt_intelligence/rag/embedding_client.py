"""Compatibility imports for the legacy Ollama embedding client path."""

from receipt_intelligence.adapters.embeddings.ollama_gateway import (
    EmbeddingClientError,
    OllamaEmbeddingClient,
)

__all__ = ["EmbeddingClientError", "OllamaEmbeddingClient"]
