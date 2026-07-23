"""Stable RAG-SQL facade backed by a LangGraph orchestrator."""

from __future__ import annotations

from receipt_intelligence.rag.candidate_resolver import CandidateResolver
from receipt_intelligence.rag_sql.answer_formatter import EvidenceBoundAnswerFormatter
from receipt_intelligence.rag_sql.executor import ReadOnlySqlExecutor
from receipt_intelligence.rag_sql.graph import build_rag_sql_graph, run_rag_sql_graph
from receipt_intelligence.rag_sql.graph_state import RagSqlGraphConfig
from receipt_intelligence.rag_sql.graph_support import RagSqlRetrievalError, SemanticRetriever
from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.planner import RagSqlPlanner
from receipt_intelligence.rag_sql.question_analyzer import RagSqlQuestionAnalyzer
from receipt_intelligence.rag_sql.validator import RagSqlValidator


class RagSqlEngine:
    """Execute one receipt question through the RAG-SQL LangGraph."""

    def __init__(
        self,
        *,
        analyzer: RagSqlQuestionAnalyzer,
        retriever: SemanticRetriever,
        resolver: CandidateResolver,
        planner: RagSqlPlanner,
        validator: RagSqlValidator,
        executor: ReadOnlySqlExecutor,
        answer_formatter: EvidenceBoundAnswerFormatter | None = None,
        retrieval_limit: int = 12,
        retrieval_minimum_score: float | None = None,
        validation_repair_count: int = 1,
        graph_config: RagSqlGraphConfig | None = None,
    ) -> None:
        if retrieval_limit <= 0 or retrieval_limit > 100:
            raise ValueError("retrieval_limit must be between 1 and 100.")
        if validation_repair_count < 0 or validation_repair_count > 3:
            raise ValueError("validation_repair_count must be between 0 and 3.")

        self.graph_config = graph_config or RagSqlGraphConfig()
        self._graph = build_rag_sql_graph(
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

    def execute(self, question: str) -> RagSqlResponse:
        normalized_question = " ".join(str(question or "").split()).strip()
        if not normalized_question:
            raise ValueError("question must not be empty.")
        return run_rag_sql_graph(
            self._graph,
            normalized_question,
            graph_config=self.graph_config,
        )


__all__ = ["RagSqlEngine", "RagSqlGraphConfig", "RagSqlRetrievalError", "SemanticRetriever"]
