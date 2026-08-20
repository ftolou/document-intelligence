"""Text-embedding provider adapters."""

from receipt_intelligence.adapters.embeddings.factory import (
    EmbeddingProviderConfig,
    build_embedding_gateway,
)
from receipt_intelligence.adapters.embeddings.ollama_gateway import OllamaEmbeddingGateway
from receipt_intelligence.adapters.embeddings.openai_gateway import OpenAIEmbeddingGateway
from receipt_intelligence.application.ports.embeddings import EmbeddingProviderError

__all__ = [
    "EmbeddingProviderConfig",
    "EmbeddingProviderError",
    "OllamaEmbeddingGateway",
    "OpenAIEmbeddingGateway",
    "build_embedding_gateway",
]
