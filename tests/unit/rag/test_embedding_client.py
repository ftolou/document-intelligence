"""Unit tests for the Ollama embedding client."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from receipt_intelligence.rag.embedding_client import (
    EmbeddingClientError,
    OllamaEmbeddingClient,
)


def _response(payload: object) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def test_embed_posts_batch_and_returns_validated_result() -> None:
    session = Mock()
    session.post.return_value = _response(
        {
            "model": "embeddinggemma",
            "embeddings": [[0.6, 0.8], [0.0, 1.0]],
            "total_duration": 120,
            "load_duration": 20,
            "prompt_eval_count": 7,
            "prompt_eval_duration": 90,
        }
    )
    client = OllamaEmbeddingClient(
        base_url="http://localhost:11434/",
        model="embeddinggemma",
        timeout_seconds=30,
        keep_alive="10m",
        session=session,
    )

    result = client.embed([" Schuhe ", "Mineralwasser"])

    assert result.count == 2
    assert result.dimension == 2
    assert result.vectors[0] == [0.6, 0.8]
    assert result.total_duration_ns == 120
    assert result.prompt_eval_duration_ns == 90
    assert len(result.ollama_calls) == 1
    assert result.ollama_calls[0].endpoint == "embed"
    assert result.ollama_calls[0].input_count == 2
    session.post.assert_called_once_with(
        "http://localhost:11434/api/embed",
        json={
            "model": "embeddinggemma",
            "input": ["Schuhe", "Mineralwasser"],
            "keep_alive": "10m",
        },
        timeout=30.0,
    )


def test_empty_batch_does_not_call_ollama() -> None:
    session = Mock()
    client = OllamaEmbeddingClient(
        base_url="http://localhost:11434",
        model="embeddinggemma",
        session=session,
    )

    result = client.embed([])

    assert result.count == 0
    assert result.dimension == 0
    session.post.assert_not_called()


def test_rejects_non_string_or_empty_inputs() -> None:
    client = OllamaEmbeddingClient(
        base_url="http://localhost:11434",
        model="embeddinggemma",
        session=Mock(),
    )

    with pytest.raises(TypeError, match="sequence of strings"):
        client.embed("Schuhe")
    with pytest.raises(TypeError, match="not a string"):
        client.embed(["Schuhe", 3])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="is empty"):
        client.embed(["   "])


def test_rejects_embedding_count_mismatch() -> None:
    session = Mock()
    session.post.return_value = _response({"model": "embeddinggemma", "embeddings": [[1.0, 0.0]]})
    client = OllamaEmbeddingClient(
        base_url="http://localhost:11434",
        model="embeddinggemma",
        session=session,
    )

    with pytest.raises(EmbeddingClientError, match="count does not match"):
        client.embed(["Schuhe", "Wasser"])


@pytest.mark.parametrize(
    "vectors, error_text",
    [
        ([[1.0, 0.0], [1.0]], "dimension"),
        ([[float("nan"), 0.0]], "non-finite"),
    ],
)
def test_rejects_invalid_vectors(vectors: list[list[float]], error_text: str) -> None:
    session = Mock()
    session.post.return_value = _response({"model": "embeddinggemma", "embeddings": vectors})
    client = OllamaEmbeddingClient(
        base_url="http://localhost:11434",
        model="embeddinggemma",
        session=session,
    )

    with pytest.raises(EmbeddingClientError, match=error_text):
        client.embed([f"text-{index}" for index in range(len(vectors))])


def test_wraps_http_failures() -> None:
    session = Mock()
    session.post.side_effect = requests.ConnectionError("offline")
    client = OllamaEmbeddingClient(
        base_url="http://localhost:11434",
        model="embeddinggemma",
        session=session,
    )

    with pytest.raises(EmbeddingClientError, match="request failed"):
        client.embed(["Schuhe"])


def test_owned_session_is_closed_idempotently_and_rejects_reuse(monkeypatch) -> None:
    session = Mock()
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = OllamaEmbeddingClient(
        base_url="http://localhost:11434",
        model="embeddinggemma",
    )

    client.close()
    client.close()

    session.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        client.embed(["Schuhe"])
