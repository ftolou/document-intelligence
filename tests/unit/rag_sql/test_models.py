from __future__ import annotations

import pytest
from pydantic import ValidationError

from receipt_intelligence.rag_sql.models import (
    QuestionAnalysisPayload,
    RagSqlPlanPayload,
    ResolvedSemanticEntity,
)


def test_question_analysis_preserves_goal_operation_and_product_entity() -> None:
    payload = QuestionAnalysisPayload.model_validate(
        {
            "schema_version": "rag_sql_question_analysis_v2",
            "status": "ready",
            "language": "de",
            "user_goal": "Berechne die gesamten Ausgaben für Schuhe.",
            "target_entity": "spending_amount",
            "requested_operation": "aggregate_sum",
            "requires_product_resolution": True,
            "entities": [
                {
                    "entity_id": "e001",
                    "search_text": "Schuhe",
                    "role": "product_filter",
                }
            ],
            "clarification_question": None,
            "reason": None,
        }
    )

    assert payload.entities[0].search_text == "Schuhe"
    assert payload.target_entity == "spending_amount"
    assert payload.requested_operation == "aggregate_sum"


def test_question_analysis_rejects_nonsequential_entity_ids() -> None:
    with pytest.raises(ValidationError, match="sequential"):
        QuestionAnalysisPayload.model_validate(
            {
                "schema_version": "rag_sql_question_analysis_v2",
                "status": "ready",
                "language": "de",
                "user_goal": "Zeige die Quittungen mit Schuhen.",
                "target_entity": "receipt",
                "requested_operation": "list",
                "requires_product_resolution": True,
                "entities": [
                    {
                        "entity_id": "e002",
                        "search_text": "Schuhe",
                        "role": "product_filter",
                    }
                ],
                "clarification_question": None,
                "reason": None,
            }
        )


def test_question_analysis_ready_requires_semantic_plan_fields() -> None:
    with pytest.raises(ValidationError, match="user_goal"):
        QuestionAnalysisPayload.model_validate(
            {
                "schema_version": "rag_sql_question_analysis_v2",
                "status": "ready",
                "language": "de",
                "requires_product_resolution": False,
                "entities": [],
                "clarification_question": None,
                "reason": None,
            }
        )


def test_sql_plan_rejects_boolean_parameter_values() -> None:
    with pytest.raises(ValidationError, match="Boolean"):
        RagSqlPlanPayload.model_validate(
            {
                "schema_version": "rag_sql_plan_v2",
                "status": "ready",
                "sql": ("SELECT COUNT(*) AS value FROM analytics_receipts WHERE receipt_id = :p"),
                "parameters": {"p": True},
                "result_shape": "scalar",
                "result_entity": "receipt_count",
                "display_columns": ["value"],
                "answer_instruction": "Report the count.",
                "clarification_question": None,
                "reason": None,
            }
        )


def test_sql_row_plan_requires_display_columns() -> None:
    with pytest.raises(ValidationError, match="display_columns"):
        RagSqlPlanPayload.model_validate(
            {
                "schema_version": "rag_sql_plan_v2",
                "status": "ready",
                "sql": "SELECT receipt_id FROM analytics_receipts LIMIT 100",
                "parameters": {},
                "result_shape": "rows",
                "result_entity": "receipt",
                "display_columns": [],
                "answer_instruction": "List receipts.",
                "clarification_question": None,
                "reason": None,
            }
        )


def test_resolved_entity_requires_selected_ids() -> None:
    with pytest.raises(ValidationError, match="selected_item_ids"):
        ResolvedSemanticEntity(
            entity_id="e001",
            search_text="Schuhe",
            status="resolved",
            selected_item_ids=[],
        )
