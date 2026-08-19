"""Unit tests for the OpenAI embedding gateway."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from receipt_intelligence.adapters.embeddings.openai_gateway import OpenAIEmbeddingGateway
from receipt_intelligence.application.ports.embeddings import EmbeddingProviderError


def _response(payload: object) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def test_embed_posts_batch_and_preserves_input_order() -> None:
    session = Mock()
    session.post.return_value = _response(
        {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                {"object": "embedding", "index": 0, "embedding": [0.6, 0.8]},
            ],
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        }
    )
    gateway = OpenAIEmbeddingGateway(
        api_key="test-key",
        model="text-embedding-3-small",
        dimensions=2,
        timeout_seconds=30,
        session=session,
    )

    result = gateway.embed([" Schuhe ", "Mineralwasser"])

    assert result.count == 2
    assert result.dimension == 2
    assert result.vectors == [[0.6, 0.8], [0.0, 1.0]]
    assert result.prompt_eval_count == 7
    assert len(result.model_calls) == 1
    assert result.model_calls[0].provider == "openai"
    assert result.model_calls[0].endpoint == "embed"
    assert result.model_calls[0].input_count == 2
    session.post.assert_called_once_with(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        json={
            "model": "text-embedding-3-small",
            "input": ["Schuhe", "Mineralwasser"],
            "encoding_format": "float",
            "dimensions": 2,
        },
        timeout=30.0,
    )


def test_empty_batch_does_not_call_openai() -> None:
    session = Mock()
    gateway = OpenAIEmbeddingGateway(
        api_key="test-key",
        model="text-embedding-3-small",
        session=session,
    )

    result = gateway.embed([])

    assert result.count == 0
    assert result.dimension == 0
    session.post.assert_not_called()


def test_rejects_invalid_inputs() -> None:
    gateway = OpenAIEmbeddingGateway(
        api_key="test-key",
        model="text-embedding-3-small",
        session=Mock(),
    )

    with pytest.raises(TypeError, match="sequence of strings"):
        gateway.embed("Schuhe")
    with pytest.raises(TypeError, match="not a string"):
        gateway.embed(["Schuhe", 3])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="is empty"):
        gateway.embed(["   "])


def test_rejects_invalid_or_duplicate_response_indexes() -> None:
    session = Mock()
    session.post.return_value = _response(
        {
            "model": "text-embedding-3-small",
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0]},
            ],
        }
    )
    gateway = OpenAIEmbeddingGateway(
        api_key="test-key",
        model="text-embedding-3-small",
        session=session,
    )

    with pytest.raises(EmbeddingProviderError, match="duplicate embedding index"):
        gateway.embed(["Schuhe", "Wasser"])


def test_rejects_embedding_count_mismatch() -> None:
    session = Mock()
    session.post.return_value = _response(
        {
            "model": "text-embedding-3-small",
            "data": [{"index": 0, "embedding": [1.0, 0.0]}],
        }
    )
    gateway = OpenAIEmbeddingGateway(
        api_key="test-key",
        model="text-embedding-3-small",
        session=session,
    )

    with pytest.raises(EmbeddingProviderError, match="count does not match"):
        gateway.embed(["Schuhe", "Wasser"])


def test_wraps_http_failures_without_exposing_configuration() -> None:
    session = Mock()
    session.post.side_effect = requests.ConnectionError("offline")
    gateway = OpenAIEmbeddingGateway(
        api_key="test-key",
        model="text-embedding-3-small",
        session=session,
    )

    with pytest.raises(EmbeddingProviderError, match="request failed"):
        gateway.embed(["Schuhe"])


def test_owned_session_is_closed_idempotently_and_rejects_reuse(monkeypatch) -> None:
    session = Mock()
    monkeypatch.setattr(requests, "Session", lambda: session)
    gateway = OpenAIEmbeddingGateway(
        api_key="test-key",
        model="text-embedding-3-small",
    )

    gateway.close()
    gateway.close()

    session.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        gateway.embed(["Schuhe"])


def test_provider_factory_selects_openai_without_exposing_secret_in_repr() -> None:
    from receipt_intelligence.adapters.embeddings import (
        EmbeddingProviderConfig,
        OpenAIEmbeddingGateway,
        build_embedding_gateway,
    )

    config = EmbeddingProviderConfig(
        provider=" OPENAI ",
        model="text-embedding-3-small",
        api_key="test-secret",
        dimensions=256,
    )
    gateway = build_embedding_gateway(config)
    try:
        assert isinstance(gateway, OpenAIEmbeddingGateway)
        assert gateway.model == "text-embedding-3-small"
        assert gateway.dimensions == 256
        assert "test-secret" not in repr(config)
    finally:
        gateway.close()


def test_provider_factory_requires_explicit_openai_credentials() -> None:
    from receipt_intelligence.adapters.embeddings import (
        EmbeddingProviderConfig,
        build_embedding_gateway,
    )

    config = EmbeddingProviderConfig(
        provider="openai",
        model="text-embedding-3-small",
    )

    with pytest.raises(ValueError, match="require an API key"):
        build_embedding_gateway(config)


def test_embedding_result_accepts_legacy_ollama_calls_field() -> None:
    from receipt_intelligence.application.ports.embeddings import EmbeddingBatchResult
    from receipt_intelligence.application.ports.llm import ModelCallMetrics

    call = ModelCallMetrics(
        provider="ollama",
        endpoint="embed",
        model="embeddinggemma",
        input_count=1,
    )
    result = EmbeddingBatchResult.model_validate(
        {
            "model": "embeddinggemma",
            "vectors": [[1.0, 0.0]],
            "dimension": 2,
            "ollama_calls": [call],
        }
    )

    assert result.model_calls == [call]
    assert result.ollama_calls == [call]
    assert "model_calls" in result.model_dump()
