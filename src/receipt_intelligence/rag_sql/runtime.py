"""Production runtime wiring for the RAG-SQL LangGraph engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from receipt_intelligence.rag.candidate_resolver import CandidateResolver, CandidateResolverConfig
from receipt_intelligence.rag_sql.answer_formatter import (
    AnswerFormatterConfig,
    EvidenceBoundAnswerFormatter,
)
from receipt_intelligence.rag.embedding_client import OllamaEmbeddingClient
from receipt_intelligence.rag.item_retriever import ItemSemanticRetriever
from receipt_intelligence.rag_sql.engine import RagSqlEngine
from receipt_intelligence.rag_sql.graph_state import RagSqlGraphConfig
from receipt_intelligence.rag_sql.executor import (
    ReadOnlySqlExecutor,
    ReadOnlySqlExecutorConfig,
)
from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.planner import RagSqlPlanner, RagSqlPlannerConfig
from receipt_intelligence.rag_sql.question_analyzer import (
    QuestionAnalyzerConfig,
    RagSqlQuestionAnalyzer,
)
from receipt_intelligence.rag_sql.validator import RagSqlValidator, SqlValidatorConfig
from receipt_intelligence.storage.connection import SQLiteConnectionFactory
from receipt_intelligence.storage.migrations import MigrationRunner


@dataclass(frozen=True)
class RagSqlRuntimeConfig:
    database_path: Path
    ollama_url: str
    embedding_model: str
    embedding_timeout_seconds: float
    embedding_keep_alive: str | None
    analyzer: QuestionAnalyzerConfig
    resolver: CandidateResolverConfig
    planner: RagSqlPlannerConfig
    answer_formatter: AnswerFormatterConfig
    retrieval_limit: int = 12
    retrieval_maximum_limit: int = 100
    retrieval_minimum_score: float | None = None
    retrieval_rrf_k: int = 60
    retrieval_vector_weight: float = 1.0
    retrieval_lexical_weight: float = 1.5
    maximum_rows: int = 100
    execution_timeout_seconds: float = 5.0
    validation_repair_count: int = 1
    graph_recursion_limit: int = 50

    def __post_init__(self) -> None:
        if self.retrieval_limit <= 0 or self.retrieval_limit > 100:
            raise ValueError("retrieval_limit must be between 1 and 100.")
        if self.retrieval_maximum_limit < self.retrieval_limit:
            raise ValueError("retrieval_maximum_limit must be >= retrieval_limit.")
        if self.maximum_rows <= 0 or self.maximum_rows > 1000:
            raise ValueError("maximum_rows must be between 1 and 1000.")
        if self.execution_timeout_seconds <= 0:
            raise ValueError("execution_timeout_seconds must be positive.")
        if self.validation_repair_count < 0 or self.validation_repair_count > 3:
            raise ValueError("validation_repair_count must be between 0 and 3.")
        if self.graph_recursion_limit < 10 or self.graph_recursion_limit > 500:
            raise ValueError("graph_recursion_limit must be between 10 and 500.")


class RagSqlRuntime:
    name = "rag_sql"

    def __init__(self, config: RagSqlRuntimeConfig) -> None:
        self.config = config

    def execute(self, question: str) -> RagSqlResponse:
        MigrationRunner(SQLiteConnectionFactory(self.config.database_path)).migrate()
        with OllamaEmbeddingClient(
            base_url=self.config.ollama_url,
            model=self.config.embedding_model,
            timeout_seconds=self.config.embedding_timeout_seconds,
            keep_alive=self.config.embedding_keep_alive,
        ) as embedding_client:
            retriever = ItemSemanticRetriever(
                database_path=self.config.database_path,
                embedding_client=embedding_client,
                maximum_limit=self.config.retrieval_maximum_limit,
                deduplicate=True,
                rrf_k=self.config.retrieval_rrf_k,
                vector_weight=self.config.retrieval_vector_weight,
                lexical_weight=self.config.retrieval_lexical_weight,
            )
            engine = RagSqlEngine(
                analyzer=RagSqlQuestionAnalyzer(self.config.analyzer),
                retriever=retriever,
                resolver=CandidateResolver(self.config.resolver),
                planner=RagSqlPlanner(self.config.planner),
                answer_formatter=EvidenceBoundAnswerFormatter(self.config.answer_formatter)
                if self.config.answer_formatter.enabled
                else None,
                validator=RagSqlValidator(
                    SqlValidatorConfig(maximum_rows=self.config.maximum_rows)
                ),
                executor=ReadOnlySqlExecutor(
                    self.config.database_path,
                    ReadOnlySqlExecutorConfig(
                        maximum_rows=self.config.maximum_rows,
                        timeout_seconds=self.config.execution_timeout_seconds,
                    ),
                ),
                retrieval_limit=self.config.retrieval_limit,
                retrieval_minimum_score=self.config.retrieval_minimum_score,
                validation_repair_count=self.config.validation_repair_count,
                graph_config=RagSqlGraphConfig(recursion_limit=self.config.graph_recursion_limit),
            )
            return engine.execute(question)


def build_rag_sql_runtime_from_settings() -> RagSqlRuntime:
    """Build the RAG-SQL runtime from application settings.

    The settings import is local so tests can import the package without
    initializing application runtime directories.
    """

    from receipt_intelligence import settings

    # One provider-side runner configuration is used for every Gemma call in
    # the RAG-SQL pipeline. Ollama may recreate a loaded model when ``num_ctx``
    # changes, even when ``keep_alive`` is set. Keeping both values identical
    # allows analysis, resolution, and planning to reuse the same resident
    # runner.
    shared_num_ctx = settings.RAG_SQL_LLM_NUM_CTX
    keep_alive = (
        settings.RAG_SQL_LLM_KEEP_ALIVE
        or settings.RAG_SQL_KEEP_ALIVE
        or settings.OLLAMA_KEEP_ALIVE
        or None
    )
    return RagSqlRuntime(
        RagSqlRuntimeConfig(
            database_path=settings.RECEIPT_DB_PATH,
            ollama_url=settings.OLLAMA_URL,
            embedding_model=settings.RAG_EMBEDDING_MODEL,
            embedding_timeout_seconds=settings.RAG_EMBEDDING_TIMEOUT_SECONDS,
            embedding_keep_alive=settings.RAG_EMBEDDING_KEEP_ALIVE or None,
            analyzer=QuestionAnalyzerConfig(
                enabled=settings.RAG_SQL_ENABLED,
                ollama_url=settings.OLLAMA_URL,
                model=settings.RAG_SQL_ANALYZER_MODEL,
                num_ctx=shared_num_ctx,
                num_predict=settings.RAG_SQL_ANALYZER_NUM_PREDICT,
                timeout_seconds=settings.RAG_SQL_ANALYZER_TIMEOUT_SECONDS,
                retry_count=settings.RAG_SQL_ANALYZER_RETRY_COUNT,
                format_json=settings.RAG_SQL_FORMAT_JSON,
                keep_alive=keep_alive,
                maximum_entities=settings.RAG_SQL_MAX_ENTITIES,
            ),
            resolver=CandidateResolverConfig(
                enabled=settings.RAG_CANDIDATE_RESOLVER_ENABLED,
                ollama_url=settings.OLLAMA_URL,
                model=settings.RAG_CANDIDATE_MODEL,
                num_ctx=shared_num_ctx,
                num_predict=settings.RAG_CANDIDATE_NUM_PREDICT,
                timeout_seconds=settings.RAG_CANDIDATE_TIMEOUT_SECONDS,
                retry_count=settings.RAG_CANDIDATE_RETRY_COUNT,
                format_json=settings.RAG_CANDIDATE_FORMAT_JSON,
                keep_alive=keep_alive,
                maximum_candidates=settings.RAG_CANDIDATE_MAX_CANDIDATES,
            ),
            answer_formatter=AnswerFormatterConfig(
                enabled=settings.RAG_SQL_ANSWER_FORMATTER_ENABLED,
                ollama_url=settings.OLLAMA_URL,
                model=settings.RAG_SQL_ANSWER_FORMATTER_MODEL,
                num_ctx=shared_num_ctx,
                num_predict=settings.RAG_SQL_ANSWER_FORMATTER_NUM_PREDICT,
                timeout_seconds=settings.RAG_SQL_ANSWER_FORMATTER_TIMEOUT_SECONDS,
                retry_count=settings.RAG_SQL_ANSWER_FORMATTER_RETRY_COUNT,
                format_json=settings.RAG_SQL_FORMAT_JSON,
                keep_alive=keep_alive,
                maximum_rows=settings.RAG_SQL_MAX_ROWS,
            ),
            planner=RagSqlPlannerConfig(
                enabled=settings.RAG_SQL_ENABLED,
                ollama_url=settings.OLLAMA_URL,
                model=settings.RAG_SQL_PLANNER_MODEL,
                num_ctx=shared_num_ctx,
                num_predict=settings.RAG_SQL_PLANNER_NUM_PREDICT,
                timeout_seconds=settings.RAG_SQL_PLANNER_TIMEOUT_SECONDS,
                retry_count=settings.RAG_SQL_PLANNER_RETRY_COUNT,
                format_json=settings.RAG_SQL_FORMAT_JSON,
                keep_alive=keep_alive,
                maximum_rows=settings.RAG_SQL_MAX_ROWS,
            ),
            retrieval_limit=settings.RAG_SQL_RETRIEVAL_LIMIT,
            retrieval_maximum_limit=settings.RAG_RETRIEVAL_MAX_LIMIT,
            retrieval_minimum_score=settings.RAG_RETRIEVAL_MINIMUM_SCORE,
            retrieval_rrf_k=settings.RAG_RETRIEVAL_RRF_K,
            retrieval_vector_weight=settings.RAG_RETRIEVAL_VECTOR_WEIGHT,
            retrieval_lexical_weight=settings.RAG_RETRIEVAL_LEXICAL_WEIGHT,
            maximum_rows=settings.RAG_SQL_MAX_ROWS,
            execution_timeout_seconds=settings.RAG_SQL_EXECUTION_TIMEOUT_SECONDS,
            validation_repair_count=settings.RAG_SQL_VALIDATION_REPAIR_COUNT,
            graph_recursion_limit=settings.RAG_SQL_GRAPH_RECURSION_LIMIT,
        )
    )


__all__ = [
    "RagSqlRuntime",
    "RagSqlRuntimeConfig",
    "build_rag_sql_runtime_from_settings",
]
