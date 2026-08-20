from __future__ import annotations

from types import SimpleNamespace

from receipt_intelligence.runtime.embedding_composition import (
    build_embedding_provider_config_from_settings,
)


def _settings(**overrides):
    values = {
        "RAG_EMBEDDING_PROVIDER": "ollama",
        "RAG_EMBEDDING_MODEL": "embeddinggemma:latest",
        "RAG_EMBEDDING_BASE_URL": "http://localhost:11434",
        "RAG_EMBEDDING_DIMENSIONS": None,
        "RAG_EMBEDDING_TIMEOUT_SECONDS": 120.0,
        "RAG_EMBEDDING_KEEP_ALIVE": "30m",
        "OPENAI_API_KEY": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_builds_ollama_embedding_provider_config() -> None:
    config = build_embedding_provider_config_from_settings(_settings())

    assert config.provider == "ollama"
    assert config.model == "embeddinggemma:latest"
    assert config.base_url == "http://localhost:11434"
    assert config.api_key is None
    assert config.dimensions is None
    assert config.keep_alive == "30m"


def test_builds_openai_embedding_provider_config_without_ollama_settings() -> None:
    config = build_embedding_provider_config_from_settings(
        _settings(
            RAG_EMBEDDING_PROVIDER="openai",
            RAG_EMBEDDING_MODEL="text-embedding-3-small",
            RAG_EMBEDDING_BASE_URL=None,
            RAG_EMBEDDING_DIMENSIONS=512,
            RAG_EMBEDDING_KEEP_ALIVE="ignored",
            OPENAI_API_KEY="test-key",
        )
    )

    assert config.provider == "openai"
    assert config.model == "text-embedding-3-small"
    assert config.base_url is None
    assert config.api_key == "test-key"
    assert config.dimensions == 512
    assert config.keep_alive is None


def test_provider_name_is_normalized_before_credentials_are_selected() -> None:
    config = build_embedding_provider_config_from_settings(
        _settings(
            RAG_EMBEDDING_PROVIDER=" OpenAI ",
            RAG_EMBEDDING_MODEL="text-embedding-3-small",
            RAG_EMBEDDING_BASE_URL=None,
            OPENAI_API_KEY="test-key",
        )
    )

    assert config.provider == "openai"
    assert config.api_key == "test-key"
