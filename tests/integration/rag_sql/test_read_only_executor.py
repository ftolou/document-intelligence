from __future__ import annotations

from pathlib import Path

import pytest

from receipt_intelligence.rag_sql.executor import ReadOnlySqlExecutor, SqlExecutionError
from receipt_intelligence.rag_sql.models import ValidatedSqlPlan
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def _insert_receipt(
    db: ReceiptDatabase,
    *,
    job_id: str,
    approved: bool,
    total: float,
    item_name: str,
    item_total: float,
    parser_item_type: str = "item",
) -> int:
    with db.connect() as connection:
        receipt_cursor = connection.execute(
            """
            INSERT INTO receipts(
                job_id, merchant_name, merchant_normalized, receipt_date,
                currency, grand_total, review_status, approved_receipt_path,
                raw_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                "Test Merchant",
                "test merchant",
                "2026-07-01",
                "EUR",
                total,
                "approved" if approved else "pending",
                f"/approved/{job_id}.json" if approved else None,
                "{}",
                "2026-07-01T00:00:00+00:00",
                "2026-07-01T00:00:00+00:00",
            ),
        )
        receipt_id = int(receipt_cursor.lastrowid)
        item_cursor = connection.execute(
            """
            INSERT INTO receipt_items(
                receipt_id, item_index, raw_name, normalized_name,
                parser_item_type, line_total, embedding_text, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                0,
                item_name,
                item_name.casefold(),
                parser_item_type,
                item_total,
                item_name.casefold(),
                "{}",
            ),
        )
        item_id = int(item_cursor.lastrowid)
        connection.commit()
    return item_id


def _plan(sql: str, parameters: dict[str, object]) -> ValidatedSqlPlan:
    return ValidatedSqlPlan(
        sql=sql,
        parameters=parameters,
        result_shape="scalar",
        result_entity="spending_amount",
        display_columns=["value", "currency"],
        answer_instruction="Report result.",
        referenced_objects=["analytics_purchase_items"],
        referenced_functions=["sum"],
        placeholder_names=sorted(parameters),
    )


def test_executor_reads_only_approved_purchase_view(tmp_path: Path) -> None:
    db = ReceiptDatabase(tmp_path / "executor.db")
    approved_item = _insert_receipt(
        db,
        job_id="approved",
        approved=True,
        total=20.0,
        item_name="HS-Halbschuhe",
        item_total=12.5,
    )
    _insert_receipt(
        db,
        job_id="pending",
        approved=False,
        total=30.0,
        item_name="Other shoe",
        item_total=30.0,
    )
    _insert_receipt(
        db,
        job_id="discount",
        approved=True,
        total=5.0,
        item_name="Discount",
        item_total=-2.0,
        parser_item_type="discount",
    )

    result = ReadOnlySqlExecutor(db.db_path).execute(
        _plan(
            "SELECT ROUND(SUM(line_total), 2) AS value, currency "
            "FROM analytics_purchase_items WHERE item_id = :item_id GROUP BY currency",
            {"item_id": approved_item},
        )
    )

    assert result.rows == [{"value": 12.5, "currency": "EUR"}]


def test_executor_authorizer_denies_direct_storage_table_access(tmp_path: Path) -> None:
    db = ReceiptDatabase(tmp_path / "authorizer.db")
    _insert_receipt(
        db,
        job_id="approved",
        approved=True,
        total=20.0,
        item_name="HS-Halbschuhe",
        item_total=12.5,
    )
    unsafe = ValidatedSqlPlan(
        sql="SELECT COUNT(*) AS value FROM receipts",
        parameters={},
        result_shape="scalar",
        result_entity="receipt_count",
        display_columns=["value"],
        answer_instruction="Report result.",
        referenced_objects=["receipts"],
        referenced_functions=["count"],
        placeholder_names=[],
    )

    with pytest.raises(SqlExecutionError, match="not authorized|denied"):
        ReadOnlySqlExecutor(db.db_path).execute(unsafe)
