from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_ask_ui_exposes_rag_sql_langgraph_engine() -> None:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert "RAG-SQL with LangGraph" in html
    assert 'id="askReceiptsEngineBadge"' in html
    assert "query-engine-summary" in html
    assert "body: JSON.stringify({ question: question.trim(), limit: 25 })" in javascript


def test_query_diagnostics_describe_graph_and_sql_stages() -> None:
    javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "Execution stages" in javascript
    assert "Resolved product entities" in javascript
    assert "Validated read-only SQL" in javascript
