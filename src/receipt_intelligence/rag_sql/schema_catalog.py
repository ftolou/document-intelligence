"""Versioned static schema and business-rule catalog for RAG-SQL planning.

Only curated analytics views are exposed to the model. The views enforce the
approved-receipt and purchased-item scope before the LLM-generated SELECT is
executed, so the planner never needs direct access to storage tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

SCHEMA_CATALOG_VERSION = "receipt_analytics_schema_v2"

ALLOWED_ANALYTICS_OBJECTS: frozenset[str] = frozenset(
    {"analytics_receipts", "analytics_purchase_items"}
)

VIEW_COLUMNS: dict[str, frozenset[str]] = {
    "analytics_receipts": frozenset(
        {
            "receipt_id",
            "job_id",
            "merchant_name",
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
        }
    ),
    "analytics_purchase_items": frozenset(
        {
            "item_id",
            "receipt_id",
            "job_id",
            "item_index",
            "merchant_name",
            "merchant",
            "receipt_date",
            "receipt_month",
            "currency",
            "description",
            "normalized_name",
            "category",
            "category_key",
            "category_reason",
            "semantic_description",
            "parser_item_type",
            "quantity",
            "unit",
            "unit_price",
            "original_price",
            "discount_amount",
            "line_total",
            "tax_code",
            "confidence",
        }
    ),
}

# SQLite authorizer must see through the views. Direct reads of these tables are
# still denied; reads are allowed only when SQLite reports an allowed view as the
# authorizing source.
VIEW_BASE_TABLES: dict[str, frozenset[str]] = {
    "analytics_receipts": frozenset({"receipts", "receipt_items"}),
    "analytics_purchase_items": frozenset({"receipts", "receipt_items"}),
}

ALLOWED_SQL_FUNCTIONS: frozenset[str] = frozenset(
    {
        "abs",
        "avg",
        "cast",
        "coalesce",
        "count",
        "date",
        "datetime",
        "glob",
        "ifnull",
        "julianday",
        "like",
        "length",
        "lower",
        "ltrim",
        "max",
        "min",
        "nullif",
        "printf",
        "replace",
        "round",
        "rtrim",
        "strftime",
        "substr",
        "sum",
        "time",
        "total",
        "trim",
        "upper",
    }
)


@dataclass(frozen=True)
class StaticSchemaCatalog:
    version: str = SCHEMA_CATALOG_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.version,
            "objects": {
                "analytics_receipts": {
                    "grain": "one row per approved receipt",
                    "columns": sorted(VIEW_COLUMNS["analytics_receipts"]),
                    "rules": [
                        "Use grand_total for complete receipt spending.",
                        "item_count counts purchased product positions only.",
                        "receipt_month is YYYY-MM only when receipt_date is ISO formatted; otherwise it is NULL.",
                    ],
                },
                "analytics_purchase_items": {
                    "grain": "one row per approved purchased-item occurrence",
                    "columns": sorted(VIEW_COLUMNS["analytics_purchase_items"]),
                    "rules": [
                        "Use line_total for spending on products or product concepts.",
                        "item_id equals receipt_items.id and is the identifier produced by product RAG.",
                        "Discount, fee, deposit, refund, subtotal, and total lines are excluded by the view.",
                        "semantic_description and category_reason are reviewed item metadata for grounded descriptive answers.",
                        "merchant and merchant_name identify the seller of the receipt; they are never product-brand fields.",
                    ],
                },
            },
            "relationships": [
                "analytics_purchase_items.receipt_id = analytics_receipts.receipt_id"
            ],
            "business_rules": [
                "Query only the two listed analytics views; never query storage tables directly.",
                "Apply every resolved typed filter with its supplied protected named parameters.",
                "Product filters constrain analytics_purchase_items.item_id; merchant filters constrain merchant; category filters constrain category; date filters constrain receipt_date.",
                "For receipt lookup by resolved product IDs, join purchase items to receipts and return DISTINCT receipt rows rather than aggregating unless the user requested an aggregate.",
                "Do not text-match description, category, merchant, payment method, or currency after the corresponding filter has been resolved.",
                "Use analytics_receipts.grand_total for whole-receipt totals and analytics_purchase_items.line_total for product totals.",
                "Do not sum receipt totals after joining to item rows because that duplicates receipt amounts.",
                "For monetary results, retain or group by currency rather than silently mixing currencies.",
                "Use bound parameters for every user-derived value.",
                "For describe_product, identify_product_type, and identify_brand, query reviewed purchase-item metadata by resolved item_id.",
                "Never infer a product brand from merchant or merchant_name: the merchant is the seller, not the brand.",
                "If reviewed metadata does not support a descriptive answer, return the rows unchanged so the application can report insufficient information.",
            ],
            "allowed_functions": sorted(ALLOWED_SQL_FUNCTIONS),
        }

    def render_for_prompt(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)


DEFAULT_SCHEMA_CATALOG = StaticSchemaCatalog()

__all__ = [
    "ALLOWED_ANALYTICS_OBJECTS",
    "ALLOWED_SQL_FUNCTIONS",
    "DEFAULT_SCHEMA_CATALOG",
    "SCHEMA_CATALOG_VERSION",
    "StaticSchemaCatalog",
    "VIEW_BASE_TABLES",
    "VIEW_COLUMNS",
]
