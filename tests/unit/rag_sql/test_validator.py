from __future__ import annotations

import pytest

from receipt_intelligence.rag_sql.models import RagSqlPlanResult
from receipt_intelligence.rag_sql.validator import RagSqlValidator, SqlValidationError


def _plan(
    sql: str,
    parameters: dict[str, object] | None = None,
    shape: str = "scalar",
    result_entity: str | None = None,
) -> RagSqlPlanResult:
    return RagSqlPlanResult.model_validate(
        {
            "schema_version": "rag_sql_plan_v2",
            "status": "ready",
            "sql": sql,
            "parameters": parameters or {},
            "result_shape": shape,
            "result_entity": result_entity or ("receipt" if shape in {"row", "rows"} else "result"),
            "display_columns": ["receipt_id"] if shape in {"row", "rows"} else ["value"],
            "answer_instruction": "Report the result.",
            "clarification_question": None,
            "reason": None,
            "model": "test",
            "attempts": 1,
            "duration_ms": 1,
        }
    )


def test_validator_accepts_parameterized_product_sum() -> None:
    plan = _plan(
        "SELECT ROUND(SUM(line_total), 2) AS value, currency "
        "FROM analytics_purchase_items "
        "WHERE item_id IN (:e001_item_0, :e001_item_1) GROUP BY currency",
        {"e001_item_0": 84, "e001_item_1": 126},
    )

    validated = RagSqlValidator().validate(
        plan,
        protected_parameters={"e001_item_0": 84, "e001_item_1": 126},
    )

    assert validated.referenced_objects == ["analytics_purchase_items"]
    assert validated.placeholder_names == ["e001_item_0", "e001_item_1"]


@pytest.mark.parametrize(
    "sql, message",
    [
        ("DELETE FROM analytics_receipts", "SELECT or WITH"),
        ("SELECT * FROM receipts", "storage object"),
        ("SELECT * FROM analytics_receipts; SELECT 1", "Exactly one"),
        ("SELECT * FROM analytics_receipts -- comment", "comments"),
        ("SELECT random() AS value FROM analytics_receipts", "non-allowlisted function"),
    ],
)
def test_validator_rejects_unsafe_sql(sql: str, message: str) -> None:
    with pytest.raises(SqlValidationError, match=message):
        RagSqlValidator().validate(_plan(sql))


def test_validator_requires_exact_named_parameter_set() -> None:
    with pytest.raises(SqlValidationError, match="match exactly"):
        RagSqlValidator().validate(
            _plan(
                "SELECT COUNT(*) AS value FROM analytics_receipts WHERE merchant = :merchant",
                {"other": "rewe"},
            )
        )


def test_validator_requires_limit_for_rows() -> None:
    with pytest.raises(SqlValidationError, match="requires a literal LIMIT"):
        RagSqlValidator().validate(_plan("SELECT receipt_id FROM analytics_receipts", shape="rows"))


def test_validator_rejects_merchant_as_product_brand() -> None:
    with pytest.raises(SqlValidationError, match="seller"):
        RagSqlValidator().validate(
            _plan(
                "SELECT merchant AS brand FROM analytics_purchase_items LIMIT 100",
                shape="rows",
                result_entity="product_brand",
            )
        )


def test_validator_enforces_merchant_parameter_column_binding() -> None:
    from receipt_intelligence.rag_sql.models import ResolvedQueryFilter

    resolved = ResolvedQueryFilter(
        filter_id="f001",
        field="merchant",
        operator="matches",
        original_value="ARAL",
        status="resolved",
        resolved_values=["aral"],
    )
    plan = _plan(
        "SELECT COUNT(*) AS value FROM analytics_purchase_items WHERE merchant = :f001_merchant_0",
        {"f001_merchant_0": "aral"},
    )

    validated = RagSqlValidator().validate(
        plan,
        protected_parameters={"f001_merchant_0": "aral"},
        resolved_filters=[resolved],
    )

    assert validated.placeholder_names == ["f001_merchant_0"]


def test_validator_rejects_merchant_parameter_bound_to_product_description() -> None:
    from receipt_intelligence.rag_sql.models import ResolvedQueryFilter

    resolved = ResolvedQueryFilter(
        filter_id="f001",
        field="merchant",
        operator="matches",
        original_value="ARAL",
        status="resolved",
        resolved_values=["aral"],
    )
    plan = _plan(
        "SELECT COUNT(*) AS value FROM analytics_purchase_items "
        "WHERE description = :f001_merchant_0",
        {"f001_merchant_0": "aral"},
    )

    with pytest.raises(SqlValidationError, match="must constrain merchant"):
        RagSqlValidator().validate(
            plan,
            protected_parameters={"f001_merchant_0": "aral"},
            resolved_filters=[resolved],
        )


def test_validator_enforces_between_filter_operator() -> None:
    from receipt_intelligence.rag_sql.models import ResolvedQueryFilter

    resolved = ResolvedQueryFilter(
        filter_id="f001",
        field="purchase_date",
        operator="between",
        original_value=["2026-01-01", "2026-12-31"],
        status="resolved",
        resolved_values=["2026-01-01", "2026-12-31"],
    )
    parameters = {
        "f001_date_0": "2026-01-01",
        "f001_date_1": "2026-12-31",
    }
    plan = _plan(
        "SELECT COUNT(*) AS value FROM analytics_receipts "
        "WHERE receipt_date BETWEEN :f001_date_0 AND :f001_date_1",
        parameters,
    )

    validated = RagSqlValidator().validate(
        plan,
        protected_parameters=parameters,
        resolved_filters=[resolved],
    )

    assert validated.placeholder_names == ["f001_date_0", "f001_date_1"]


def test_validator_rejects_between_bounds_split_across_amount_columns() -> None:
    from receipt_intelligence.rag_sql.models import ResolvedQueryFilter

    resolved = ResolvedQueryFilter(
        filter_id="f001",
        field="amount",
        operator="between",
        original_value=[10.0, 20.0],
        status="resolved",
        resolved_values=[10.0, 20.0],
    )
    parameters = {
        "f001_amount_0": 10.0,
        "f001_amount_1": 20.0,
    }
    plan = _plan(
        "SELECT COUNT(*) AS value FROM analytics_purchase_items "
        "WHERE line_total >= :f001_amount_0 AND grand_total <= :f001_amount_1",
        parameters,
    )

    with pytest.raises(SqlValidationError, match="must constrain grand_total, line_total"):
        RagSqlValidator().validate(
            plan,
            protected_parameters=parameters,
            resolved_filters=[resolved],
        )
