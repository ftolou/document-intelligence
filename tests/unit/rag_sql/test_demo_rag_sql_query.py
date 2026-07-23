from __future__ import annotations

import ast
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_demo_runtime_config_includes_hybrid_answer_formatter() -> None:
    script_path = _project_root() / "scripts" / "demo_rag_sql_query.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))

    runtime_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RagSqlRuntimeConfig"
    ]
    assert len(runtime_calls) == 1

    keyword_names = {keyword.arg for keyword in runtime_calls[0].keywords}
    assert "answer_formatter" in keyword_names

    formatter_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AnswerFormatterConfig"
    ]
    assert len(formatter_calls) == 1


def test_demo_uses_one_shared_llm_residency_configuration() -> None:
    script_path = _project_root() / "scripts" / "demo_rag_sql_query.py"
    source = script_path.read_text(encoding="utf-8")

    assert "shared_num_ctx = settings.RAG_SQL_LLM_NUM_CTX" in source
    assert source.count("num_ctx=shared_num_ctx") == 4
    assert "settings.RAG_SQL_LLM_KEEP_ALIVE" in source
