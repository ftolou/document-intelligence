"""SQLite catalog for canonical low-cardinality query-filter values."""

from __future__ import annotations

from pathlib import Path

from receipt_intelligence.rag_sql.filter_definitions import FilterField
from receipt_intelligence.storage.connection import SQLiteConnectionFactory


class SQLiteFilterValueCatalog:
    def __init__(self, database_path: Path | str) -> None:
        self.connections = SQLiteConnectionFactory(database_path)

    def values(self, field: FilterField) -> list[str]:
        query = {
            "merchant": (
                "SELECT DISTINCT merchant AS value FROM analytics_receipts "
                "WHERE merchant IS NOT NULL AND trim(merchant) <> '' ORDER BY merchant"
            ),
            "category": (
                "SELECT DISTINCT category AS value FROM analytics_purchase_items "
                "WHERE category IS NOT NULL AND trim(category) <> '' ORDER BY category"
            ),
            "payment_method": (
                "SELECT DISTINCT payment_method AS value FROM analytics_receipts "
                "WHERE payment_method IS NOT NULL AND trim(payment_method) <> '' "
                "ORDER BY payment_method"
            ),
            "currency": (
                "SELECT DISTINCT currency AS value FROM analytics_receipts "
                "WHERE currency IS NOT NULL AND trim(currency) <> '' ORDER BY currency"
            ),
        }.get(field)
        if query is None:
            return []
        with self.connections.connect_read_only() as connection:
            return [str(row["value"]) for row in connection.execute(query).fetchall()]


__all__ = ["SQLiteFilterValueCatalog"]
