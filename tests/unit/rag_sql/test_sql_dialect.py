from __future__ import annotations

import json

import pytest

from receipt_intelligence.rag_sql.models import QuestionAnalysisResult, RagSqlPlanResult
from receipt_intelligence.rag_sql.planner import RagSqlPlanner, RagSqlPlannerConfig
from receipt_intelligence.rag_sql.schema_catalog import schema_catalog_for_dialect
from receipt_intelligence.rag_sql.sql_dialect import get_sql_dialect_profile
from receipt_intelligence.rag_sql.validator import (
    RagSqlValidator,
    SqlValidationError,
    SqlValidatorConfig,
)


def _analysis() -> QuestionAnalysisResult:
    return QuestionAnalysisResult(
        status="ready",
        language="en",
        user_goal="Count approved receipts.",
        target_entity="receipt",
        requested_operation="count",
        requires_product_resolution=False,
        entities=[],
        model="test",
        attempts=1,
    )


def _plan(sql: str) -> RagSqlPlanResult:
    return RagSqlPlanResult.model_validate(
        {
            "schema_version": "rag_sql_plan_v2",
            "status": "ready",
            "sql": sql,
            "parameters": {},
            "result_shape": "scalar",
            "result_entity": "receipt_count",
            "display_columns": ["value"],
            "answer_instruction": "Report the count.",
            "clarification_question": None,
            "reason": None,
            "model": "test",
            "attempts": 1,
            "duration_ms": 1,
        }
    )


def test_postgresql_profile_drives_planner_prompt_and_schema_catalog() -> None:
    prompts: list[str] = []

    def generate(**kwargs: object) -> str:
        prompts.append(str(kwargs["prompt"]))
        return json.dumps(
            {
                "schema_version": "rag_sql_plan_v2",
                "status": "ready",
                "sql": "SELECT COUNT(*) AS value FROM analytics_receipts",
                "parameters": {},
                "result_shape": "scalar",
                "result_entity": "receipt_count",
                "display_columns": ["value"],
                "answer_instruction": "Report the count.",
                "clarification_question": None,
                "reason": None,
            }
        )

    planner = RagSqlPlanner(
        RagSqlPlannerConfig(sql_dialect="postgresql", retry_count=0),
        schema_catalog=schema_catalog_for_dialect("postgresql"),
        generate=generate,
    )
    result = planner.plan(
        "How many receipts do I have?",
        analysis=_analysis(),
        resolved_entities=[],
        protected_parameters={},
    )

    assert result.status == "ready"
    assert prompts
    assert "safe read-only PostgreSQL query" in prompts[0]
    assert '"sql_dialect": "postgresql"' in prompts[0]
    assert "strftime" not in schema_catalog_for_dialect("postgresql").as_dict()["allowed_functions"]


def test_planner_rejects_dialect_catalog_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        RagSqlPlanner(
            RagSqlPlannerConfig(sql_dialect="postgresql"),
            schema_catalog=schema_catalog_for_dialect("sqlite"),
        )


def test_postgresql_validator_profile_rejects_sqlite_only_function() -> None:
    profile = get_sql_dialect_profile("postgresql")
    validator = RagSqlValidator(SqlValidatorConfig(allowed_functions=profile.allowed_functions))

    with pytest.raises(SqlValidationError, match="non-allowlisted function"):
        validator.validate(
            _plan("SELECT strftime('%Y', receipt_date) AS value FROM analytics_receipts")
        )

    validated = validator.validate(_plan("SELECT COUNT(*) AS value FROM analytics_receipts"))
    assert validated.referenced_functions == ["count"]


def test_unknown_sql_dialect_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported SQL dialect"):
        get_sql_dialect_profile("mysql")
