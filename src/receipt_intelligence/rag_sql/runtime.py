"""Long-lived production composition for the RAG-SQL engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from receipt_intelligence.adapters.llm import OllamaGateway
from receipt_intelligence.adapters.storage.sqlite.analytical_query import (
    SQLiteAnalyticalQueryRepository,
)
from receipt_intelligence.adapters.storage.sqlite.semantic_search import (
    SQLiteSemanticSearchRepository,
)
from receipt_intelligence.rag.candidate_resolver import CandidateResolver, CandidateResolverConfig
from receipt_intelligence.rag.embedding_client import OllamaEmbeddingClient
from receipt_intelligence.rag.item_retriever import EmbeddingClient, ItemSemanticRetriever
from receipt_intelligence.rag_sql.answer_formatter import (
    AnswerFormatterConfig,
    EvidenceBoundAnswerFormatter,
)
from receipt_intelligence.rag_sql.engine import RagSqlEngine
from receipt_intelligence.rag_sql.executor import (
    ReadOnlySqlExecutor,
    ReadOnlySqlExecutorConfig,
)
from receipt_intelligence.rag_sql.graph_state import RagSqlGraphConfig
from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.orchestration.contracts import RagSqlOrchestratorFactory
from receipt_intelligence.rag_sql.planner import RagSqlPlanner, RagSqlPlannerConfig
from receipt_intelligence.rag_sql.question_analyzer import (
    QuestionAnalyzerConfig,
    RagSqlQuestionAnalyzer,
)
from receipt_intelligence.rag_sql.validator import RagSqlValidator, SqlValidatorConfig


class RagSqlExecutor(Protocol):
    def execute(self, question: str) -> RagSqlResponse: ...


class Closeable(Protocol):
    def close(self) -> None: ...


RagSqlEngineFactory = Callable[["RagSqlRuntimeConfig", EmbeddingClient], RagSqlExecutor]


@dataclass(frozen=True, slots=True)
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
    """Own one process-scoped RAG-SQL object graph and its resources."""

    name = "rag_sql"

    def __init__(
        self,
        config: RagSqlRuntimeConfig,
        *,
        engine: RagSqlExecutor | None = None,
        embedding_client: EmbeddingClient | None = None,
        engine_factory: RagSqlEngineFactory | None = None,
    ) -> None:
        if engine is not None and (embedding_client is not None or engine_factory is not None):
            raise ValueError(
                "engine cannot be combined with embedding_client or engine_factory."
            )

        self.config = config
        self._closed = False
        self._owned_resources: list[Closeable] = []

        if engine is not None:
            self._engine = engine
            return

        resolved_embedding_client = embedding_client or OllamaEmbeddingClient(
            base_url=config.ollama_url,
            model=config.embedding_model,
            timeout_seconds=config.embedding_timeout_seconds,
            keep_alive=config.embedding_keep_alive,
        )
        if embedding_client is None and hasattr(resolved_embedding_client, "close"):
            self._owned_resources.append(resolved_embedding_client)  # type: ignore[arg-type]

        resolved_factory = engine_factory or build_rag_sql_engine
        try:
            self._engine = resolved_factory(config, resolved_embedding_client)
        except Exception:
            self.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def orchestrator_name(self) -> str:
        return str(getattr(self._engine, "orchestrator_name", "unknown"))

    @property
    def orchestrator_version(self) -> str:
        return str(getattr(self._engine, "orchestrator_version", "unknown"))

    def execute(self, question: str) -> RagSqlResponse:
        if self._closed:
            raise RuntimeError("RAG-SQL runtime is closed.")
        return self._engine.execute(question)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in reversed(self._owned_resources):
            resource.close()
        self._owned_resources.clear()

    def __enter__(self) -> RagSqlRuntime:
        if self._closed:
            raise RuntimeError("RAG-SQL runtime is closed.")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def build_rag_sql_engine(
    config: RagSqlRuntimeConfig,
    embedding_client: EmbeddingClient,
    *,
    orchestrator_factory: RagSqlOrchestratorFactory | None = None,
) -> RagSqlEngine:
    """Compose one reusable engine from stable adapters and policies."""

    llm_gateway = OllamaGateway(config.ollama_url)
    retriever = ItemSemanticRetriever(
        repository=SQLiteSemanticSearchRepository(config.database_path),
        embedding_client=embedding_client,
        maximum_limit=config.retrieval_maximum_limit,
        deduplicate=True,
        rrf_k=config.retrieval_rrf_k,
        vector_weight=config.retrieval_vector_weight,
        lexical_weight=config.retrieval_lexical_weight,
    )
    return RagSqlEngine(
        analyzer=RagSqlQuestionAnalyzer(config.analyzer, llm_gateway=llm_gateway),
        retriever=retriever,
        resolver=CandidateResolver(config.resolver, llm_gateway=llm_gateway),
        planner=RagSqlPlanner(config.planner, llm_gateway=llm_gateway),
        answer_formatter=(
            EvidenceBoundAnswerFormatter(config.answer_formatter, llm_gateway=llm_gateway)
            if config.answer_formatter.enabled
            else None
        ),
        validator=RagSqlValidator(SqlValidatorConfig(maximum_rows=config.maximum_rows)),
        executor=ReadOnlySqlExecutor(
            SQLiteAnalyticalQueryRepository(config.database_path),
            ReadOnlySqlExecutorConfig(
                maximum_rows=config.maximum_rows,
                timeout_seconds=config.execution_timeout_seconds,
            ),
        ),
        retrieval_limit=config.retrieval_limit,
        retrieval_minimum_score=config.retrieval_minimum_score,
        validation_repair_count=config.validation_repair_count,
        graph_config=RagSqlGraphConfig(recursion_limit=config.graph_recursion_limit),
        orchestrator_factory=orchestrator_factory,
    )


def build_rag_sql_runtime_config_from_settings() -> RagSqlRuntimeConfig:
    """Translate application settings without composing optional graph adapters."""

    from receipt_intelligence import settings

    # Ollama may recreate a loaded model whenever ``num_ctx`` changes. One
    # provider-side configuration lets all RAG-SQL LLM stages reuse the same
    # resident runner.
    shared_num_ctx = settings.RAG_SQL_LLM_NUM_CTX
    keep_alive = (
        settings.RAG_SQL_LLM_KEEP_ALIVE
        or settings.RAG_SQL_KEEP_ALIVE
        or settings.OLLAMA_KEEP_ALIVE
        or None
    )
    return RagSqlRuntimeConfig(
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


def build_rag_sql_runtime_from_settings() -> RagSqlRuntime:
    """Compose the process-scoped RAG-SQL runtime once at application startup."""

    return RagSqlRuntime(build_rag_sql_runtime_config_from_settings())


__all__ = [
    "RagSqlEngineFactory",
    "RagSqlExecutor",
    "RagSqlRuntime",
    "RagSqlRuntimeConfig",
    "build_rag_sql_engine",
    "build_rag_sql_runtime_config_from_settings",
    "build_rag_sql_runtime_from_settings",
]
