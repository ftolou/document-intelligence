from __future__ import annotations

import pytest
from pydantic import ValidationError

from receipt_intelligence.rag_sql.models import (
    QueryFilter,
    QuestionAnalysisPayload,
    QuestionAnalysisResult,
    RagSqlPlanPayload,
    ResolvedQueryFilter,
    ResolvedSemanticEntity,
    SemanticEntity,
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


def test_question_analysis_accepts_generic_merchant_filter() -> None:
    payload = QuestionAnalysisPayload.model_validate(
        {
            "schema_version": "rag_sql_question_analysis_v3",
            "status": "ready",
            "language": "en",
            "user_goal": "List purchases made at ARAL.",
            "target_entity": "purchase_item",
            "requested_operation": "list",
            "filters": [
                {
                    "filter_id": "f001",
                    "field": "merchant",
                    "operator": "matches",
                    "value": "ARAL",
                }
            ],
            "clarification_question": None,
            "reason": None,
        }
    )

    assert payload.filters == [
        QueryFilter(
            filter_id="f001",
            field="merchant",
            operator="matches",
            value="ARAL",
        )
    ]
    assert payload.requires_product_resolution is False
    assert payload.entities == []


def test_question_analysis_migrates_legacy_model_instances() -> None:
    payload = QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Berechne die Ausgaben für Schuhe.",
        target_entity="spending_amount",
        requested_operation="aggregate_sum",
        requires_product_resolution=True,
        entities=[SemanticEntity(entity_id="e001", search_text="Schuhe")],
        model="test",
        attempts=1,
    )

    assert payload.schema_version == "rag_sql_question_analysis_v2"
    assert payload.filters[0].filter_id == "e001"
    assert payload.filters[0].field == "product"
    assert payload.filters[0].value == "Schuhe"


def test_query_filter_rejects_operator_not_supported_by_field() -> None:
    with pytest.raises(ValidationError, match="not valid for merchant"):
        QueryFilter(
            filter_id="f001",
            field="merchant",
            operator="greater_than",
            value="ARAL",
        )


def test_not_found_resolved_filter_rejects_clarification_question() -> None:
    with pytest.raises(ValidationError, match="clarification_question"):
        ResolvedQueryFilter(
            filter_id="f001",
            field="merchant",
            operator="matches",
            original_value="ARAL",
            status="not_found",
            clarification_question="Which ARAL?",
        )
