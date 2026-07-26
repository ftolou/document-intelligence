"""Provider-neutral orchestration contracts for the RAG-SQL pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from receipt_intelligence.rag_sql.answer_formatter import EvidenceBoundAnswerFormatter
from receipt_intelligence.rag_sql.executor import ReadOnlySqlExecutor
from receipt_intelligence.rag_sql.filter_resolution import QueryFilterResolver
from receipt_intelligence.rag_sql.graph_state import RagSqlGraphConfig
from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.planner import RagSqlPlanner
from receipt_intelligence.rag_sql.question_analyzer import RagSqlQuestionAnalyzer
from receipt_intelligence.rag_sql.validator import RagSqlValidator

RAG_SQL_GRAPH_VERSION = "rag_sql_graph_v2"


@dataclass(frozen=True, slots=True)
class RagSqlComponents:
    """Stable dependencies consumed by an orchestration adapter."""

    analyzer: RagSqlQuestionAnalyzer
    filter_resolver: QueryFilterResolver
    planner: RagSqlPlanner
    validator: RagSqlValidator
    executor: ReadOnlySqlExecutor
    answer_formatter: EvidenceBoundAnswerFormatter | None
    validation_repair_count: int


class RagSqlOrchestrator(Protocol):
    """Execute a normalized question through one orchestration implementation."""

    name: str
    version: str

    def execute(self, question: str) -> RagSqlResponse: ...


class RagSqlOrchestratorFactory(Protocol):
    """Compile an orchestrator once from stable RAG-SQL components."""

    def build(
        self,
        components: RagSqlComponents,
        *,
        graph_config: RagSqlGraphConfig,
    ) -> RagSqlOrchestrator: ...


__all__ = [
    "RAG_SQL_GRAPH_VERSION",
    "RagSqlComponents",
    "RagSqlOrchestrator",
    "RagSqlOrchestratorFactory",
]
