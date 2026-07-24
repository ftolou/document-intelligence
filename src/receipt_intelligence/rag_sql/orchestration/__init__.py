"""RAG-SQL orchestration contracts.

The LangGraph implementation is intentionally not imported here. Consumers that
select that adapter should import ``receipt_intelligence.rag_sql.orchestration.langgraph``
from the composition boundary.
"""

from receipt_intelligence.rag_sql.orchestration.contracts import (
    RAG_SQL_GRAPH_VERSION,
    RagSqlComponents,
    RagSqlOrchestrator,
    RagSqlOrchestratorFactory,
)

__all__ = [
    "RAG_SQL_GRAPH_VERSION",
    "RagSqlComponents",
    "RagSqlOrchestrator",
    "RagSqlOrchestratorFactory",
]
