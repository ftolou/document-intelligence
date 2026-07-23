from __future__ import annotations

import json

import pytest

from receipt_intelligence.rag_sql.models import (
    QuestionAnalysisResult,
    RagSqlPlanResult,
    ResolvedSemanticEntity,
    SemanticEntity,
)
from receipt_intelligence.rag_sql.planner import (
    RagSqlPlanner,
    RagSqlPlannerConfig,
    RagSqlPlanningError,
    build_protected_item_parameters,
)


def _analysis() -> QuestionAnalysisResult:
    return QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Berechne die gesamten Ausgaben für Schuhe.",
        target_entity="spending_amount",
        requested_operation="aggregate_sum",
        requires_product_resolution=True,
        entities=[SemanticEntity(entity_id="e001", search_text="Schuhe")],
        model="test",
        attempts=1,
    )


def _receipt_analysis() -> QuestionAnalysisResult:
    return QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Zeige die Quittung oder Quittungen mit Vittel.",
        target_entity="receipt",
        requested_operation="list",
        requires_product_resolution=True,
        entities=[SemanticEntity(entity_id="e001", search_text="Vittel")],
        model="test",
        attempts=1,
    )


def _resolved(search_text: str = "Schuhe") -> list[ResolvedSemanticEntity]:
    return [
        ResolvedSemanticEntity(
            entity_id="e001",
            search_text=search_text,
            status="resolved",
            selected_item_ids=[84, 126],
        )
    ]


def test_planner_preserves_app_owned_item_parameters() -> None:
    protected = build_protected_item_parameters(_resolved())

    def generate(**kwargs: object) -> str:
        prompt = str(kwargs["prompt"])
        assert '"e001_item_0": 84' in prompt
        assert '"requested_operation": "aggregate_sum"' in prompt
        return json.dumps(
            {
                "schema_version": "rag_sql_plan_v2",
                "status": "ready",
                "sql": (
                    "SELECT ROUND(SUM(line_total), 2) AS value, currency "
                    "FROM analytics_purchase_items "
                    "WHERE item_id IN (:e001_item_0, :e001_item_1) "
                    "GROUP BY currency"
                ),
                "parameters": protected,
                "result_shape": "scalar",
                "result_entity": "spending_amount",
                "display_columns": ["value", "currency"],
                "answer_instruction": "Report spending on resolved shoes.",
                "clarification_question": None,
                "reason": None,
            }
        )

    result = RagSqlPlanner(RagSqlPlannerConfig(retry_count=0), generate=generate).plan(
        "Wie viel habe ich für Schuhe ausgegeben?",
        analysis=_analysis(),
        resolved_entities=_resolved(),
        protected_parameters=protected,
    )

    assert result.parameters == protected
    assert result.status == "ready"
    assert result.result_entity == "spending_amount"


def test_planner_preserves_receipt_lookup_semantics() -> None:
    resolved = _resolved("Vittel")
    protected = build_protected_item_parameters(resolved)

    def generate(**kwargs: object) -> str:
        prompt = str(kwargs["prompt"])
        assert "Zeige mir die Quittung mit Vittel" in prompt
        assert '"target_entity": "receipt"' in prompt
        assert '"requested_operation": "list"' in prompt
        assert "must return receipt rows, not a spending total" in prompt
        return json.dumps(
            {
                "schema_version": "rag_sql_plan_v2",
                "status": "ready",
                "sql": (
                    "SELECT DISTINCT R.receipt_id, R.receipt_date, R.merchant, "
                    "R.grand_total, R.currency FROM analytics_receipts AS R "
                    "JOIN analytics_purchase_items AS I "
                    "ON I.receipt_id = R.receipt_id "
                    "WHERE I.item_id IN (:e001_item_0, :e001_item_1) "
                    "ORDER BY R.receipt_date DESC, R.receipt_id DESC LIMIT 100"
                ),
                "parameters": protected,
                "result_shape": "rows",
                "result_entity": "receipt",
                "display_columns": [
                    "receipt_id",
                    "receipt_date",
                    "merchant",
                    "grand_total",
                    "currency",
                ],
                "answer_instruction": "List the receipts containing Vittel.",
                "clarification_question": None,
                "reason": None,
            }
        )

    result = RagSqlPlanner(RagSqlPlannerConfig(retry_count=0), generate=generate).plan(
        "Zeige mir die Quittung mit Vittel.",
        analysis=_receipt_analysis(),
        resolved_entities=resolved,
        protected_parameters=protected,
    )

    assert result.result_shape == "rows"
    assert result.result_entity == "receipt"
    assert "SUM(" not in (result.sql or "").upper()
    assert result.display_columns[0] == "receipt_id"


def test_planner_rejects_modified_protected_parameter() -> None:
    protected = build_protected_item_parameters(_resolved())

    def generate(**_: object) -> str:
        changed = dict(protected)
        changed["e001_item_0"] = 999
        return json.dumps(
            {
                "schema_version": "rag_sql_plan_v2",
                "status": "ready",
                "sql": (
                    "SELECT 1 AS value FROM analytics_purchase_items WHERE item_id=:e001_item_0"
                ),
                "parameters": changed,
                "result_shape": "scalar",
                "result_entity": "spending_amount",
                "display_columns": ["value"],
                "answer_instruction": "Report result.",
                "clarification_question": None,
                "reason": None,
            }
        )

    with pytest.raises(RagSqlPlanningError, match="changed"):
        RagSqlPlanner(RagSqlPlannerConfig(retry_count=0), generate=generate).plan(
            "Schuhe",
            analysis=_analysis(),
            resolved_entities=_resolved(),
            protected_parameters=protected,
        )


def test_planner_validation_repair_includes_exact_error_and_previous_plan() -> None:
    protected = build_protected_item_parameters(_resolved())
    previous_plan = {
        "schema_version": "rag_sql_plan_v2",
        "status": "ready",
        "sql": (
            "SELECT SUM(line_total) AS value, currency "
            "FROM analytics_purchase_items "
            "WHERE item_id IN (:e001_item_0, :e001_item_1) "
            "GROUP BY currency"
        ),
        "parameters": protected,
        "result_shape": "grouped_rows",
        "result_entity": "spending_amount",
        "display_columns": ["value", "currency"],
        "answer_instruction": "Report spending.",
        "clarification_question": None,
        "reason": None,
        "model": "test",
        "attempts": 1,
        "duration_ms": 0.0,
    }
    seen_prompts: list[str] = []

    def generate(**kwargs: object) -> str:
        prompt = str(kwargs["prompt"])
        seen_prompts.append(prompt)
        assert "grouped_rows result_shape requires a literal LIMIT" in prompt
        assert previous_plan["sql"] in prompt
        assert "complete replacement JSON plan" in prompt
        assert "LIMIT 100 unless the original question explicitly requests fewer groups" in prompt
        assert "Never use LIMIT 1 merely to satisfy validation" in prompt
        return json.dumps(
            {
                "schema_version": previous_plan["schema_version"],
                "status": previous_plan["status"],
                "sql": f"{previous_plan['sql']} ORDER BY currency LIMIT 100",
                "parameters": previous_plan["parameters"],
                "result_shape": previous_plan["result_shape"],
                "result_entity": previous_plan["result_entity"],
                "display_columns": previous_plan["display_columns"],
                "answer_instruction": previous_plan["answer_instruction"],
                "clarification_question": previous_plan["clarification_question"],
                "reason": previous_plan["reason"],
            }
        )

    planner = RagSqlPlanner(
        RagSqlPlannerConfig(retry_count=0),
        generate=generate,
    )
    planner_result = RagSqlPlanResult.model_validate(previous_plan)
    result = planner.repair_after_validation_failure(
        "Wie viel habe ich für Schuhe ausgegeben?",
        analysis=_analysis(),
        resolved_entities=_resolved(),
        protected_parameters=protected,
        previous_plan=planner_result,
        validation_error=(
            "SqlValidationError: grouped_rows result_shape requires a literal LIMIT."
        ),
    )

    assert result.sql is not None and result.sql.endswith("LIMIT 100")
    assert result.parameters == protected
    assert result.attempts == 1
    assert seen_prompts
    assert planner_result.status == "ready"


def test_planner_requires_complete_validation_repair_context() -> None:
    protected = build_protected_item_parameters(_resolved())

    with pytest.raises(ValueError, match="both be provided"):
        RagSqlPlanner(RagSqlPlannerConfig(retry_count=0)).plan(
            "Schuhe",
            analysis=_analysis(),
            resolved_entities=_resolved(),
            protected_parameters=protected,
            validation_error="missing limit",
        )


def test_planner_retries_count_metadata_with_exact_contract_error() -> None:
    analysis = QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Zähle, wie oft Zahnpasta gekauft wurde.",
        target_entity="purchase_item",
        requested_operation="count",
        requires_product_resolution=True,
        entities=[SemanticEntity(entity_id="e001", search_text="Zahnpasta")],
        model="test",
        attempts=1,
    )
    resolved = [
        ResolvedSemanticEntity(
            entity_id="e001",
            search_text="Zahnpasta",
            status="resolved",
            selected_item_ids=[43],
        )
    ]
    protected = build_protected_item_parameters(resolved)
    invalid = {
        "schema_version": "rag_sql_plan_v2",
        "status": "ready",
        "sql": (
            "SELECT COUNT(T1.item_id) AS value FROM analytics_purchase_items AS T1 "
            "WHERE T1.item_id = :e001_item_0"
        ),
        "parameters": protected,
        "result_shape": "scalar",
        "result_entity": "count of purchased items",
        "display_columns": ["COUNT(T1.item_id)"],
        "answer_instruction": "Report the count.",
        "clarification_question": None,
        "reason": None,
    }
    valid = {
        **invalid,
        "result_entity": "purchase_count",
        "display_columns": ["value"],
    }
    prompts: list[str] = []
    outputs = iter([json.dumps(invalid), json.dumps(valid)])

    def generate(**kwargs: object) -> str:
        prompts.append(str(kwargs["prompt"]))
        return next(outputs)

    result = RagSqlPlanner(
        RagSqlPlannerConfig(retry_count=1),
        generate=generate,
    ).plan(
        "Wie oft habe ich Zahnpasta gekauft?",
        analysis=analysis,
        resolved_entities=resolved,
        protected_parameters=protected,
    )

    assert result.result_entity == "purchase_count"
    assert result.display_columns == ["value"]
    assert result.attempts == 2
    assert len(prompts) == 2
    assert "Exact validation error" in prompts[1]
    assert "count of purchased items" in prompts[1]
    assert "COUNT(T1.item_id)" in prompts[1]
    assert "display_columns must contain only SELECT output names or aliases" in prompts[1]


def test_planner_prompt_requires_currency_for_monetary_results() -> None:
    protected = build_protected_item_parameters(_resolved())

    def generate(**kwargs: object) -> str:
        prompt = str(kwargs["prompt"])
        assert "Every monetary result must return both currency" in prompt
        assert "Do not return a monetary scalar containing only value" in prompt
        return json.dumps(
            {
                "schema_version": "rag_sql_plan_v2",
                "status": "ready",
                "sql": (
                    "SELECT currency, SUM(line_total) AS value "
                    "FROM analytics_purchase_items "
                    "WHERE item_id IN (:e001_item_0, :e001_item_1) "
                    "GROUP BY currency ORDER BY currency LIMIT 100"
                ),
                "parameters": protected,
                "result_shape": "grouped_rows",
                "result_entity": "spending_amount",
                "display_columns": ["currency", "value"],
                "answer_instruction": "Report spending with currency.",
                "clarification_question": None,
                "reason": None,
            }
        )

    result = RagSqlPlanner(
        RagSqlPlannerConfig(retry_count=0),
        generate=generate,
    ).plan(
        "Wie viel habe ich für Schuhe ausgegeben?",
        analysis=_analysis(),
        resolved_entities=_resolved(),
        protected_parameters=protected,
    )

    assert result.result_shape == "grouped_rows"
    assert result.display_columns == ["currency", "value"]


def test_planner_prompt_forbids_treating_merchant_as_brand() -> None:
    analysis = QuestionAnalysisResult(
        status="ready",
        language="en",
        user_goal="Identify the brand of Vittel.",
        target_entity="product_brand",
        requested_operation="identify_brand",
        requires_product_resolution=True,
        entities=[SemanticEntity(entity_id="e001", search_text="Vittel")],
        model="test",
        attempts=1,
    )
    resolved = _resolved("Vittel")
    protected = build_protected_item_parameters(resolved)

    def generate(**kwargs: object) -> str:
        prompt = str(kwargs["prompt"])
        assert "merchant and merchant_name are seller fields" in prompt
        assert "Never select, alias, or interpret either one as a product brand" in prompt
        return json.dumps(
            {
                "schema_version": "rag_sql_plan_v2",
                "status": "ready",
                "sql": (
                    "SELECT DISTINCT item_id, description, normalized_name, semantic_description, "
                    "category, category_reason FROM analytics_purchase_items "
                    "WHERE item_id IN (:e001_item_0, :e001_item_1) ORDER BY item_id LIMIT 100"
                ),
                "parameters": protected,
                "result_shape": "rows",
                "result_entity": "product_brand",
                "display_columns": [
                    "item_id",
                    "description",
                    "normalized_name",
                    "semantic_description",
                    "category",
                    "category_reason",
                ],
                "answer_instruction": "Return reviewed product metadata for deterministic brand identification.",
                "clarification_question": None,
                "reason": None,
            }
        )

    result = RagSqlPlanner(RagSqlPlannerConfig(retry_count=0), generate=generate).plan(
        "Which brand is Vittel?",
        analysis=analysis,
        resolved_entities=resolved,
        protected_parameters=protected,
    )
    assert "merchant" not in (result.sql or "").casefold()
