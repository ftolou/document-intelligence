"""Stable RAG-SQL facade independent of a concrete graph library."""

from __future__ import annotations

from receipt_intelligence.rag.candidate_resolver import CandidateResolver
from receipt_intelligence.rag_sql.answer_formatter import EvidenceBoundAnswerFormatter
from receipt_intelligence.rag_sql.executor import ReadOnlySqlExecutor
from receipt_intelligence.rag_sql.filter_resolution import (
    FilterValueCatalog,
    QueryFilterResolverRegistry,
)
from receipt_intelligence.rag_sql.graph_state import RagSqlGraphConfig
from receipt_intelligence.rag_sql.graph_support import RagSqlRetrievalError, SemanticRetriever
from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.orchestration.contracts import (
    RagSqlComponents,
    RagSqlOrchestrator,
    RagSqlOrchestratorFactory,
)
from receipt_intelligence.rag_sql.planner import RagSqlPlanner
from receipt_intelligence.rag_sql.question_analyzer import RagSqlQuestionAnalyzer
from receipt_intelligence.rag_sql.validator import RagSqlValidator


class RagSqlEngine:
    """Execute receipt questions through one compiled orchestration adapter."""

    def __init__(
        self,
        *,
        analyzer: RagSqlQuestionAnalyzer,
        retriever: SemanticRetriever,
        resolver: CandidateResolver,
        filter_catalog: FilterValueCatalog | None = None,
        planner: RagSqlPlanner,
        validator: RagSqlValidator,
        executor: ReadOnlySqlExecutor,
        answer_formatter: EvidenceBoundAnswerFormatter | None = None,
        retrieval_limit: int = 12,
        retrieval_minimum_score: float | None = None,
        validation_repair_count: int = 1,
        graph_config: RagSqlGraphConfig | None = None,
        orchestrator_factory: RagSqlOrchestratorFactory | None = None,
    ) -> None:
        if retrieval_limit <= 0 or retrieval_limit > 100:
            raise ValueError("retrieval_limit must be between 1 and 100.")
        if validation_repair_count < 0 or validation_repair_count > 3:
            raise ValueError("validation_repair_count must be between 0 and 3.")
        if planner.sql_dialect.name != validator.sql_dialect.name:
            raise ValueError(
                "planner and validator SQL dialects must match: "
                f"planner={planner.sql_dialect.name!r}, "
                f"validator={validator.sql_dialect.name!r}."
            )

        resolved_factory = orchestrator_factory or _default_orchestrator_factory()
        self.graph_config = graph_config or RagSqlGraphConfig()
        filter_resolver = QueryFilterResolverRegistry(
            retriever=retriever,
            product_resolver=resolver,
            catalog=filter_catalog,
            retrieval_limit=retrieval_limit,
            retrieval_minimum_score=retrieval_minimum_score,
        )
        self._orchestrator: RagSqlOrchestrator = resolved_factory.build(
            RagSqlComponents(
                analyzer=analyzer,
                filter_resolver=filter_resolver,
                planner=planner,
                validator=validator,
                executor=executor,
                answer_formatter=answer_formatter,
                validation_repair_count=validation_repair_count,
            ),
            graph_config=self.graph_config,
        )

    @property
    def orchestrator_name(self) -> str:
        return self._orchestrator.name

    @property
    def orchestrator_version(self) -> str:
        return self._orchestrator.version

    def execute(self, question: str) -> RagSqlResponse:
        normalized_question = " ".join(str(question or "").split()).strip()
        if not normalized_question:
            raise ValueError("question must not be empty.")
        return self._orchestrator.execute(normalized_question)


def _default_orchestrator_factory() -> RagSqlOrchestratorFactory:
    """Load LangGraph only when an engine is actually composed."""

    from receipt_intelligence.rag_sql.orchestration.langgraph import (
        LangGraphRagSqlOrchestratorFactory,
    )

    return LangGraphRagSqlOrchestratorFactory()


__all__ = [
    "RagSqlEngine",
    "RagSqlGraphConfig",
    "RagSqlRetrievalError",
    "SemanticRetriever",
]
