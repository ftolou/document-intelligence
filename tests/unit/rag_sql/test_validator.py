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
