"""Compatibility facade for the LangGraph RAG-SQL adapter.

New composition code should depend on the provider-neutral orchestration
contracts and import the LangGraph adapter explicitly. These functions remain
for existing callers without importing ``langgraph`` at module-import time.
"""

from __future__ import annotations

from typing import Any

from receipt_intelligence.rag.candidate_resolver import CandidateResolver
from receipt_intelligence.rag_sql.answer_formatter import EvidenceBoundAnswerFormatter
from receipt_intelligence.rag_sql.executor import ReadOnlySqlExecutor
from receipt_intelligence.rag_sql.graph_state import RagSqlGraphConfig, RagSqlGraphState
from receipt_intelligence.rag_sql.graph_support import SemanticRetriever
from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.orchestration.contracts import (
    RAG_SQL_GRAPH_VERSION,
    RagSqlComponents,
)
from receipt_intelligence.rag_sql.planner import RagSqlPlanner
from receipt_intelligence.rag_sql.question_analyzer import RagSqlQuestionAnalyzer
from receipt_intelligence.rag_sql.validator import RagSqlValidator


def build_rag_sql_graph(
    *,
    analyzer: RagSqlQuestionAnalyzer,
    retriever: SemanticRetriever,
    resolver: CandidateResolver,
    planner: RagSqlPlanner,
    validator: RagSqlValidator,
    executor: ReadOnlySqlExecutor,
    retrieval_limit: int,
    retrieval_minimum_score: float | None,
    validation_repair_count: int,
    answer_formatter: EvidenceBoundAnswerFormatter | None = None,
) -> Any:
    """Compile the legacy graph facade using the isolated LangGraph adapter."""

    from receipt_intelligence.rag_sql.orchestration.langgraph import (
        compile_rag_sql_graph,
    )

    return compile_rag_sql_graph(
        RagSqlComponents(
            analyzer=analyzer,
            retriever=retriever,
            resolver=resolver,
            planner=planner,
            validator=validator,
            executor=executor,
            answer_formatter=answer_formatter,
            retrieval_limit=retrieval_limit,
            retrieval_minimum_score=retrieval_minimum_score,
            validation_repair_count=validation_repair_count,
        )
    )


def run_rag_sql_graph(
    graph: Any,
    question: str,
    *,
    graph_config: RagSqlGraphConfig,
) -> RagSqlResponse:
    """Run a graph produced by :func:`build_rag_sql_graph`."""

    from receipt_intelligence.rag_sql.orchestration.langgraph import (
        run_compiled_rag_sql_graph,
    )

    return run_compiled_rag_sql_graph(
        graph,
        question,
        graph_config=graph_config,
    )


__all__ = [
    "RAG_SQL_GRAPH_VERSION",
    "RagSqlGraphConfig",
    "RagSqlGraphState",
    "build_rag_sql_graph",
    "run_rag_sql_graph",
]
