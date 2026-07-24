"""Public API for the RAG-assisted read-only SQL strategy.

Exports are resolved lazily so importing lightweight models, ports, or storage
adapters does not require the optional LangGraph runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "ANSWER_FORMAT_SCHEMA_VERSION": "receipt_intelligence.rag_sql.answer_formatter",
    "AnswerFormatterConfig": "receipt_intelligence.rag_sql.answer_formatter",
    "AnswerFormatterPayload": "receipt_intelligence.rag_sql.answer_formatter",
    "AnswerFormatterResult": "receipt_intelligence.rag_sql.answer_formatter",
    "AnswerFormattingError": "receipt_intelligence.rag_sql.answer_formatter",
    "AnswerValidationResult": "receipt_intelligence.rag_sql.answer_formatter",
    "EvidenceBoundAnswerFormatter": "receipt_intelligence.rag_sql.answer_formatter",
    "render_validated_answer": "receipt_intelligence.rag_sql.answer_formatter",
    "validate_answer_formatter_result": "receipt_intelligence.rag_sql.answer_formatter",
    "RagSqlEngine": "receipt_intelligence.rag_sql.engine",
    "RagSqlRetrievalError": "receipt_intelligence.rag_sql.engine",
    "ReadOnlySqlExecutor": "receipt_intelligence.rag_sql.executor",
    "ReadOnlySqlExecutorConfig": "receipt_intelligence.rag_sql.executor",
    "SqlExecutionError": "receipt_intelligence.rag_sql.executor",
    "RAG_SQL_GRAPH_VERSION": "receipt_intelligence.rag_sql.orchestration.contracts",
    "RagSqlGraphConfig": "receipt_intelligence.rag_sql.graph_state",
    "RagSqlGraphState": "receipt_intelligence.rag_sql.graph_state",
    "RAG_SQL_ANALYSIS_SCHEMA_VERSION": "receipt_intelligence.rag_sql.models",
    "RAG_SQL_ENGINE_VERSION": "receipt_intelligence.rag_sql.models",
    "RAG_SQL_PLAN_SCHEMA_VERSION": "receipt_intelligence.rag_sql.models",
    "QuestionAnalysisPayload": "receipt_intelligence.rag_sql.models",
    "QuestionAnalysisResult": "receipt_intelligence.rag_sql.models",
    "RagSqlPlanPayload": "receipt_intelligence.rag_sql.models",
    "RagSqlPlanResult": "receipt_intelligence.rag_sql.models",
    "RagSqlResponse": "receipt_intelligence.rag_sql.models",
    "ResolvedSemanticEntity": "receipt_intelligence.rag_sql.models",
    "SemanticEntity": "receipt_intelligence.rag_sql.models",
    "SqlExecutionResult": "receipt_intelligence.rag_sql.models",
    "ValidatedSqlPlan": "receipt_intelligence.rag_sql.models",
    "RagSqlPlanner": "receipt_intelligence.rag_sql.planner",
    "RagSqlPlannerConfig": "receipt_intelligence.rag_sql.planner",
    "RagSqlPlanningError": "receipt_intelligence.rag_sql.planner",
    "build_protected_item_parameters": "receipt_intelligence.rag_sql.planner",
    "QuestionAnalysisError": "receipt_intelligence.rag_sql.question_analyzer",
    "QuestionAnalyzerConfig": "receipt_intelligence.rag_sql.question_analyzer",
    "RagSqlQuestionAnalyzer": "receipt_intelligence.rag_sql.question_analyzer",
    "RagSqlRuntime": "receipt_intelligence.rag_sql.runtime",
    "RagSqlRuntimeConfig": "receipt_intelligence.rag_sql.runtime",
    "build_rag_sql_engine": "receipt_intelligence.rag_sql.runtime",
    "build_rag_sql_runtime_config_from_settings": "receipt_intelligence.rag_sql.runtime",
    "build_rag_sql_runtime_from_settings": "receipt_intelligence.rag_sql.runtime",
    "DEFAULT_SCHEMA_CATALOG": "receipt_intelligence.rag_sql.schema_catalog",
    "SCHEMA_CATALOG_VERSION": "receipt_intelligence.rag_sql.schema_catalog",
    "StaticSchemaCatalog": "receipt_intelligence.rag_sql.schema_catalog",
    "RagSqlValidator": "receipt_intelligence.rag_sql.validator",
    "SqlValidationError": "receipt_intelligence.rag_sql.validator",
    "SqlValidatorConfig": "receipt_intelligence.rag_sql.validator",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
