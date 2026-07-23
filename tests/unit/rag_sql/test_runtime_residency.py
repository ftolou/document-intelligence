from __future__ import annotations

from pathlib import Path

from receipt_intelligence.rag_sql.runtime import build_rag_sql_runtime_from_settings


def test_rag_sql_runtime_uses_one_resident_llm_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from receipt_intelligence import settings

    monkeypatch.setattr(settings, "RECEIPT_DB_PATH", tmp_path / "receipts.db")
    monkeypatch.setattr(settings, "RAG_SQL_LLM_NUM_CTX", 6144)
    monkeypatch.setattr(settings, "RAG_SQL_LLM_KEEP_ALIVE", "30m")
    monkeypatch.setattr(settings, "RAG_SQL_KEEP_ALIVE", "legacy-value")
    monkeypatch.setattr(settings, "OLLAMA_KEEP_ALIVE", "global-value")
    monkeypatch.setattr(settings, "RAG_EMBEDDING_KEEP_ALIVE", "30m")

    # Deliberately conflicting legacy stage settings must not fragment the
    # RAG-SQL Gemma runner.
    monkeypatch.setattr(settings, "RAG_SQL_ANALYZER_NUM_CTX", 3072)
    monkeypatch.setattr(settings, "RAG_CANDIDATE_NUM_CTX", 4096)
    monkeypatch.setattr(settings, "RAG_SQL_PLANNER_NUM_CTX", 8192)
    monkeypatch.setattr(settings, "RAG_CANDIDATE_KEEP_ALIVE", "0")

    runtime = build_rag_sql_runtime_from_settings()
    config = runtime.config

    assert config.analyzer.num_ctx == 6144
    assert config.resolver.num_ctx == 6144
    assert config.planner.num_ctx == 6144
    assert config.answer_formatter.num_ctx == 6144

    assert config.analyzer.keep_alive == "30m"
    assert config.resolver.keep_alive == "30m"
    assert config.planner.keep_alive == "30m"
    assert config.answer_formatter.keep_alive == "30m"
    assert config.embedding_keep_alive == "30m"
