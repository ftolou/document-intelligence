from __future__ import annotations

from pathlib import Path

import pytest

from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.runtime import (
    RagSqlRuntime,
    build_rag_sql_runtime_config_from_settings,
)


class _FakeEmbeddingClient:
    model = "fake-embedding"

    def __init__(self) -> None:
        self.close_count = 0

    def embed(self, texts):  # pragma: no cover - engine fake does not retrieve
        raise AssertionError(f"unexpected embedding request: {texts}")

    def close(self) -> None:
        self.close_count += 1


class _FakeEngine:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def execute(self, question: str) -> RagSqlResponse:
        self.questions.append(question)
        return RagSqlResponse(
            question=question,
            status="completed",
            answer="ok",
        )


def _config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from receipt_intelligence import settings

    monkeypatch.setattr(settings, "RECEIPT_DB_PATH", tmp_path / "receipts.db")
    return build_rag_sql_runtime_config_from_settings()


def test_runtime_composes_once_and_reuses_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import receipt_intelligence.rag_sql.runtime as runtime_module

    config = _config(monkeypatch, tmp_path)
    embedding_client = _FakeEmbeddingClient()
    engine = _FakeEngine()
    factory_calls: list[object] = []

    monkeypatch.setattr(
        runtime_module,
        "OllamaEmbeddingClient",
        lambda **_kwargs: embedding_client,
    )

    def engine_factory(received_config, received_embedding):
        factory_calls.append((received_config, received_embedding))
        return engine

    runtime = RagSqlRuntime(config, engine_factory=engine_factory)

    assert len(factory_calls) == 1
    assert runtime.execute("first").answer == "ok"
    assert runtime.execute("second").answer == "ok"
    assert len(factory_calls) == 1
    assert engine.questions == ["first", "second"]

    runtime.close()
    runtime.close()
    assert embedding_client.close_count == 1
    assert runtime.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        runtime.execute("third")


def test_runtime_closes_owned_client_when_engine_composition_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import receipt_intelligence.rag_sql.runtime as runtime_module

    config = _config(monkeypatch, tmp_path)
    embedding_client = _FakeEmbeddingClient()
    monkeypatch.setattr(
        runtime_module,
        "OllamaEmbeddingClient",
        lambda **_kwargs: embedding_client,
    )

    def failing_factory(_config, _embedding):
        raise RuntimeError("composition failed")

    with pytest.raises(RuntimeError, match="composition failed"):
        RagSqlRuntime(config, engine_factory=failing_factory)

    assert embedding_client.close_count == 1


def test_injected_engine_has_external_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(monkeypatch, tmp_path)
    engine = _FakeEngine()

    with RagSqlRuntime(config, engine=engine) as runtime:
        assert runtime.execute("question").status == "completed"

    assert runtime.closed is True
