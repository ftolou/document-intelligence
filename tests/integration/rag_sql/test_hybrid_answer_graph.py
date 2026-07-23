from __future__ import annotations

import json

from receipt_intelligence.rag_sql.answer_formatter import (
    AnswerFormatterConfig,
    EvidenceBoundAnswerFormatter,
)
from receipt_intelligence.rag_sql.engine import RagSqlEngine
from receipt_intelligence.rag_sql.models import (
    QuestionAnalysisResult,
    RagSqlPlanResult,
    SqlExecutionResult,
    ValidatedSqlPlan,
)


class FakeAnalyzer:
    def analyze(self, _question: str) -> QuestionAnalysisResult:
        return QuestionAnalysisResult(
            status="ready",
            language="en",
            user_goal="Identify the reviewed coffee brand.",
            target_entity="product_brand",
            requested_operation="identify_brand",
            requires_product_resolution=False,
            entities=[],
            model="test",
            attempts=1,
        )


class FakePlanner:
    def plan(self, *_args: object, **_kwargs: object) -> RagSqlPlanResult:
        return RagSqlPlanResult(
            status="ready",
            sql="SELECT 1",
            parameters={},
            result_shape="rows",
            result_entity="product_brand",
            display_columns=[
                "item_id",
                "description",
                "normalized_name",
                "semantic_description",
                "category",
                "category_reason",
            ],
            answer_instruction="Identify the reviewed product brand.",
            model="test",
            attempts=1,
        )


class FakeValidator:
    def validate(self, plan: RagSqlPlanResult, **_kwargs: object) -> ValidatedSqlPlan:
        return ValidatedSqlPlan(
            sql=plan.sql or "SELECT 1",
            parameters=plan.parameters,
            result_shape="rows",
            result_entity="product_brand",
            display_columns=plan.display_columns,
            answer_instruction=plan.answer_instruction or "Identify the brand.",
        )


class FakeExecutor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def execute(self, _plan: ValidatedSqlPlan) -> SqlExecutionResult:
        return SqlExecutionResult(
            columns=list(self.rows[0]) if self.rows else [],
            rows=self.rows,
            row_count=len(self.rows),
            duration_ms=1.0,
        )


class UnusedRetriever:
    pass


class UnusedResolver:
    pass


def _ambiguous_rows() -> list[dict[str, object]]:
    return [
        {
            "item_id": 175,
            "description": "STARBUCKS NESPRESSO CAPSULES",
            "normalized_name": "Starbucks Nespresso Capsules",
            "semantic_description": None,
            "category": "groceries_food",
            "category_reason": (
                "The packaging identifies Starbucks as the product brand and "
                "Nespresso as the compatible system."
            ),
        }
    ]


def _engine(
    *,
    rows: list[dict[str, object]],
    payload: dict[str, object],
    calls: list[str],
) -> RagSqlEngine:
    def generate(**_kwargs: object) -> str:
        calls.append("called")
        return json.dumps(payload)

    return RagSqlEngine(
        analyzer=FakeAnalyzer(),  # type: ignore[arg-type]
        retriever=UnusedRetriever(),  # type: ignore[arg-type]
        resolver=UnusedResolver(),  # type: ignore[arg-type]
        planner=FakePlanner(),  # type: ignore[arg-type]
        validator=FakeValidator(),  # type: ignore[arg-type]
        executor=FakeExecutor(rows),  # type: ignore[arg-type]
        answer_formatter=EvidenceBoundAnswerFormatter(
            AnswerFormatterConfig(model="test-model", retry_count=0),
            generate=generate,
        ),
    )


def test_langgraph_routes_ambiguous_evidence_through_validated_llm_fallback() -> None:
    calls: list[str] = []
    engine = _engine(
        rows=_ambiguous_rows(),
        payload={
            "schema_version": "rag_sql_answer_format_v1",
            "status": "resolved",
            "values": ["Starbucks"],
            "supporting_item_ids": [175],
            "evidence_fields": ["description", "category_reason"],
            "reason": "Starbucks is explicitly the product brand.",
        },
        calls=calls,
    )

    response = engine.execute("What was the coffee brand?")

    assert calls == ["called"]
    assert response.status == "completed"
    assert response.answer == "The brand named in the reviewed product data is “Starbucks”."
    assert response.diagnostics["graph_version"] == "rag_sql_graph_v2"
    assert response.diagnostics["answer_formatting"] == {
        "deterministic_status": "ambiguous",
        "deterministic_reason": "reviewed_evidence_requires_semantic_normalization",
        "fallback_available": True,
        "fallback_used": True,
        "validation_status": "valid",
        "supporting_item_ids": [175],
        "fallback_model": "test-model",
        "fallback_status": "resolved",
        "fallback_attempts": 1,
        "validation_reason": "all_values_supported_by_reviewed_rows",
        "evidence_fields": ["description", "category_reason"],
        "values": ["Starbucks"],
    }
    trace = [entry["node"] for entry in response.diagnostics["graph_trace"]]
    assert trace[-4:] == [
        "extract_answer",
        "format_answer_with_llm",
        "validate_formatted_answer",
        "finalize_response",
    ]


def test_invalid_llm_value_is_rejected_to_insufficient_info() -> None:
    calls: list[str] = []
    engine = _engine(
        rows=_ambiguous_rows(),
        payload={
            "schema_version": "rag_sql_answer_format_v1",
            "status": "resolved",
            "values": ["Jacobs"],
            "supporting_item_ids": [175],
            "evidence_fields": ["description", "category_reason"],
            "reason": "Unsupported invented value.",
        },
        calls=calls,
    )

    response = engine.execute("What was the coffee brand?")

    assert calls == ["called"]
    assert response.status == "insufficient_info"
    assert response.diagnostics["answer_formatting"]["validation_status"] == "invalid"
    assert response.diagnostics["answer_formatting"]["validation_reason"] == (
        "unsupported_value:Jacobs"
    )


def test_clear_deterministic_brand_skips_llm_fallback() -> None:
    calls: list[str] = []
    rows = [
        {
            "item_id": 175,
            "description": "*SENSEO CLASSIC 1",
            "normalized_name": "*SENSEO CLASSIC 1",
            "semantic_description": None,
            "category": "groceries_food",
            "category_reason": "Senseo is a brand of coffee/coffee pods.",
        }
    ]
    engine = _engine(
        rows=rows,
        payload={
            "schema_version": "rag_sql_answer_format_v1",
            "status": "resolved",
            "values": ["unused"],
            "supporting_item_ids": [175],
            "evidence_fields": ["description"],
            "reason": "unused",
        },
        calls=calls,
    )

    response = engine.execute("What was the coffee brand?")

    assert calls == []
    assert response.status == "completed"
    assert response.answer == "The brand named in the reviewed product data is “Senseo”."
    assert response.diagnostics["answer_formatting"]["fallback_used"] is False
