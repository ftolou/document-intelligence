"""Typed state and configuration for the RAG-SQL LangGraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from receipt_intelligence.rag_sql.answer_formatter import (
    AnswerFormatterResult,
    AnswerValidationResult,
)
from receipt_intelligence.rag_sql.formatter import DeterministicAnswerDecision
from receipt_intelligence.rag_sql.models import (
    QuestionAnalysisResult,
    RagSqlPlanResult,
    RagSqlResponse,
    ResolvedSemanticEntity,
    SqlExecutionResult,
    ValidatedSqlPlan,
)


@dataclass(frozen=True, slots=True)
class RagSqlGraphConfig:
    recursion_limit: int = 50

    def __post_init__(self) -> None:
        if self.recursion_limit < 10 or self.recursion_limit > 500:
            raise ValueError("recursion_limit must be between 10 and 500.")


RagSqlGraphRoute = Literal[
    "retrieve",
    "plan",
    "validate",
    "repair",
    "execute",
    "extract",
    "llm_format",
    "validate_answer",
    "finalize",
    "terminal",
    "fail",
]


class RagSqlGraphState(TypedDict, total=False):
    question: str
    started_at_perf: float
    diagnostics: dict[str, object]
    route: RagSqlGraphRoute

    analysis: QuestionAnalysisResult
    entity_index: int
    resolved_entities: list[ResolvedSemanticEntity]
    retrieval_diagnostics: list[dict[str, object]]
    protected_parameters: dict[str, int]

    plan: RagSqlPlanResult
    validated_plan: ValidatedSqlPlan
    execution: SqlExecutionResult
    deterministic_answer: DeterministicAnswerDecision
    llm_answer_result: AnswerFormatterResult
    answer_validation: AnswerValidationResult
    validation_attempt: int
    first_validation_error: str | None
    validation_error: str | None

    terminal_status: str
    terminal_answer: str
    clarification_question: str | None
    error_code: str
    exception: Exception

    final_response: RagSqlResponse
    metadata: dict[str, Any]


__all__ = ["RagSqlGraphConfig", "RagSqlGraphRoute", "RagSqlGraphState"]
