"""Read-only receipt analytics and planner context queries."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.storage.migrations import LATEST_SCHEMA_VERSION
from receipt_intelligence.storage.normalization import normalize_merchant_name
from receipt_intelligence.storage.repositories.base import BaseRepository


class AnalyticsRepository(BaseRepository):
    def receipt_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM receipts").fetchone()
            return int(row["n"])

    def item_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM receipt_items").fetchone()
            return int(row["n"])

    def summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            receipts = int(connection.execute("SELECT COUNT(*) AS n FROM receipts").fetchone()["n"])
            items = int(
                connection.execute("SELECT COUNT(*) AS n FROM receipt_items").fetchone()["n"]
            )
            merchants = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT merchant_normalized AS merchant,
                           COUNT(*) AS receipt_count,
                           ROUND(COALESCE(SUM(grand_total), 0), 2) AS total_amount
                    FROM receipts
                    GROUP BY merchant_normalized
                    ORDER BY receipt_count DESC, merchant ASC
                    LIMIT 10
                    """
                ).fetchall()
            ]
            categories = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT COALESCE(category, 'uncategorized') AS category,
                           COUNT(*) AS item_count,
                           ROUND(COALESCE(SUM(line_total), 0), 2) AS total_amount
                    FROM receipt_items
                    GROUP BY COALESCE(category, 'uncategorized')
                    ORDER BY item_count DESC, category ASC
                    LIMIT 10
                    """
                ).fetchall()
            ]
        return {
            "db_path": str(self.connections.database_path),
            "schema_version": LATEST_SCHEMA_VERSION,
            "receipt_count": receipts,
            "item_count": items,
            "top_merchants": merchants,
            "top_categories": categories,
        }

    def query_planner_context(self) -> dict[str, Any]:
        with self.connect() as connection:
            merchants = [
                row["merchant"]
                for row in connection.execute(
                    """
                    SELECT DISTINCT merchant_normalized AS merchant
                    FROM receipts
                    WHERE merchant_normalized IS NOT NULL
                      AND merchant_normalized <> ''
                    ORDER BY merchant
                    LIMIT 100
                    """
                ).fetchall()
            ]
            categories = [
                row["category"]
                for row in connection.execute(
                    """
                    SELECT DISTINCT category
                    FROM receipt_items
                    WHERE category IS NOT NULL AND category <> ''
                    ORDER BY category
                    LIMIT 200
                    """
                ).fetchall()
            ]
            date_row = connection.execute(
                "SELECT MIN(receipt_date) AS date_min, MAX(receipt_date) AS date_max FROM receipts"
            ).fetchone()
            counts = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM receipts) AS receipt_count,
                    (SELECT COUNT(*) FROM receipt_items) AS item_count
                """
            ).fetchone()
        return {
            "receipt_count": int(counts["receipt_count"] or 0),
            "item_count": int(counts["item_count"] or 0),
            "date_min": date_row["date_min"],
            "date_max": date_row["date_max"],
            "merchants": merchants,
            "categories": categories,
            "analytical_dsl": {
                "entities": ["items", "receipts"],
                "item_fields": [
                    "id",
                    "receipt_id",
                    "job_id",
                    "merchant",
                    "receipt_date",
                    "receipt_month",
                    "currency",
                    "description",
                    "normalized_name",
                    "category",
                    "category_key",
                    "quantity",
                    "unit",
                    "unit_price",
                    "original_price",
                    "discount_amount",
                    "line_total",
                    "tax_code",
                    "confidence",
                ],
                "receipt_fields": [
                    "id",
                    "job_id",
                    "merchant",
                    "receipt_date",
                    "receipt_month",
                    "receipt_time",
                    "currency",
                    "subtotal",
                    "tax_total",
                    "grand_total",
                    "paid_total",
                    "payment_method",
                    "item_count",
                ],
                "filter_operators": [
                    "eq",
                    "neq",
                    "contains",
                    "gt",
                    "gte",
                    "lt",
                    "lte",
                    "between",
                    "in",
                    "not_in",
                    "is_null",
                    "not_null",
                ],
                "aggregate_functions": ["count", "sum", "average", "minimum", "maximum"],
                "result_shapes": ["scalar", "row", "rows", "grouped_rows"],
            },
        }

    def list_receipts_filtered(
        self,
        *,
        merchant: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 25,
        sort_by: str = "date_desc",
    ) -> list[dict[str, Any]]:
        merchant_normalized = normalize_merchant_name(merchant) if merchant else None
        clauses: list[str] = []
        parameters: list[Any] = []
        if merchant_normalized:
            clauses.append("r.merchant_normalized = ?")
            parameters.append(merchant_normalized)
        if date_from:
            clauses.append("r.receipt_date >= ?")
            parameters.append(date_from)
        if date_to:
            clauses.append("r.receipt_date <= ?")
            parameters.append(date_to)
        where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
        order_sql = {
            "amount_desc": "r.grand_total DESC, r.receipt_date DESC",
            "name_asc": "r.merchant_normalized ASC, r.receipt_date DESC",
        }.get(sort_by, "r.receipt_date DESC, r.receipt_time DESC")
        sql = f"""
            SELECT r.id AS receipt_db_id, r.job_id, r.merchant_name,
                   r.merchant_normalized, r.receipt_date, r.receipt_time,
                   r.currency, r.subtotal, r.tax_total, r.grand_total,
                   r.paid_total, r.payment_method, r.review_status,
                   r.approved_receipt_path, r.image_path, COUNT(i.id) AS item_count
            FROM receipts r
            LEFT JOIN receipt_items i ON i.receipt_id = r.id
            {where_sql}
            GROUP BY r.id
            ORDER BY {order_sql}
            LIMIT ?
        """
        parameters.append(max(1, min(10000, int(limit))))
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, parameters).fetchall()]

    def aggregate_receipts(
        self,
        *,
        merchant: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        aggregation: str = "sum",
        metric: str = "grand_total",
    ) -> dict[str, Any]:
        merchant_normalized = normalize_merchant_name(merchant) if merchant else None
        clauses: list[str] = []
        parameters: list[Any] = []
        if merchant_normalized:
            clauses.append("merchant_normalized = ?")
            parameters.append(merchant_normalized)
        if date_from:
            clauses.append("receipt_date >= ?")
            parameters.append(date_from)
        if date_to:
            clauses.append("receipt_date <= ?")
            parameters.append(date_to)
        where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
        if metric == "receipt_count" or aggregation == "count":
            expression = "COUNT(*)"
        elif aggregation == "average":
            expression = "AVG(grand_total)"
        else:
            expression = "SUM(grand_total)"
        sql = f"""
            SELECT {expression} AS value,
                   COUNT(*) AS receipt_count,
                   COUNT(DISTINCT currency) AS currency_count,
                   MIN(currency) AS currency
            FROM receipts
            {where_sql}
        """
        with self.connect() as connection:
            row = connection.execute(sql, parameters).fetchone()
        value = row["value"]
        if value is not None and metric != "receipt_count" and aggregation != "count":
            value = round(float(value), 2)
        elif value is not None:
            value = int(value)
        return {
            "value": value or 0,
            "aggregation": aggregation,
            "metric": metric,
            "receipt_count": int(row["receipt_count"] or 0),
            "currency": row["currency"] if int(row["currency_count"] or 0) <= 1 else "MIXED",
        }

    def group_receipts(
        self,
        *,
        group_by: str,
        merchant: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        aggregation: str = "sum",
        metric: str = "grand_total",
        limit: int = 25,
        sort_by: str = "amount_desc",
    ) -> list[dict[str, Any]]:
        if group_by not in {"merchant", "month"}:
            raise ValueError(f"Unsupported receipt grouping: {group_by}")
        merchant_normalized = normalize_merchant_name(merchant) if merchant else None
        clauses: list[str] = []
        parameters: list[Any] = []
        if merchant_normalized:
            clauses.append("merchant_normalized = ?")
            parameters.append(merchant_normalized)
        if date_from:
            clauses.append("receipt_date >= ?")
            parameters.append(date_from)
        if date_to:
            clauses.append("receipt_date <= ?")
            parameters.append(date_to)
        where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
        key_expression = (
            "COALESCE(merchant_normalized, 'unknown')"
            if group_by == "merchant"
            else "COALESCE(substr(receipt_date, 1, 7), 'unknown')"
        )
        if metric == "receipt_count" or aggregation == "count":
            value_expression = "COUNT(*)"
        elif aggregation == "average":
            value_expression = "AVG(grand_total)"
        else:
            value_expression = "SUM(grand_total)"
        order_expression = (
            "value DESC" if sort_by in {"amount_desc", "count_desc"} else "group_key ASC"
        )
        sql = f"""
            SELECT {key_expression} AS group_key,
                   {value_expression} AS value,
                   COUNT(*) AS receipt_count,
                   COUNT(DISTINCT currency) AS currency_count,
                   MIN(currency) AS currency
            FROM receipts
            {where_sql}
            GROUP BY {key_expression}
            ORDER BY {order_expression}
            LIMIT ?
        """
        parameters.append(max(1, min(1000, int(limit))))
        with self.connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            value = row["value"]
            if value is not None and metric != "receipt_count" and aggregation != "count":
                value = round(float(value), 2)
            elif value is not None:
                value = int(value)
            result.append(
                {
                    "group_key": row["group_key"],
                    "value": value or 0,
                    "aggregation": aggregation,
                    "metric": metric,
                    "receipt_count": int(row["receipt_count"] or 0),
                    "currency": row["currency"]
                    if int(row["currency_count"] or 0) <= 1
                    else "MIXED",
                }
            )
        return result
