"""Receipt-item persistence helpers."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from receipt_intelligence.storage.normalization import (
    as_float,
    as_str,
    build_item_embedding_text,
    category_from_item,
    extract_item_description,
    first_present,
    parser_item_type_from_item,
)
from receipt_intelligence.storage.repositories.base import BaseRepository, fts_available


class ItemRepository(BaseRepository):
    def insert_items(
        self,
        connection: sqlite3.Connection,
        receipt_id: int,
        receipt: dict[str, Any],
        merchant_name: str | None,
        merchant_normalized: str | None,
    ) -> int:
        items = receipt.get("items") if isinstance(receipt.get("items"), list) else []
        currency = as_str(receipt.get("currency")) or "EUR"
        receipt_date = as_str(receipt.get("date"))
        has_fts = fts_available(connection)

        inserted = 0
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            description = extract_item_description(item)
            if not description.strip():
                continue
            normalized_name = as_str(
                first_present(
                    item.get("normalized_name"),
                    item.get("product_description"),
                    item.get("name"),
                    description,
                )
            )
            category = category_from_item(item, description)
            line_total = as_float(
                first_present(item.get("line_total"), item.get("total"), item.get("amount"))
            )
            embedding_text = build_item_embedding_text(
                merchant_name=merchant_name,
                merchant_normalized=merchant_normalized,
                receipt_date=receipt_date,
                item=item,
                description=description,
                normalized_name=normalized_name,
                category=category,
                line_total=line_total,
                currency=currency,
            )
            cursor = connection.execute(
                """
                INSERT INTO receipt_items(
                    receipt_id, item_index, raw_name, normalized_name, category,
                    parser_item_type, category_group, category_key, category_reason,
                    semantic_description, quantity, unit, unit_price, original_price,
                    discount_amount, line_total, tax_code, vat_rate, confidence,
                    review_status, embedding_text, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    index,
                    description,
                    normalized_name,
                    category,
                    parser_item_type_from_item(item),
                    as_str(item.get("category_group")),
                    as_str(item.get("category_key")),
                    as_str(item.get("category_reason")),
                    as_str(item.get("semantic_description")),
                    as_float(item.get("quantity")),
                    as_str(item.get("unit")),
                    as_float(item.get("unit_price")),
                    as_float(
                        first_present(item.get("original_price"), item.get("gross_unit_price"))
                    ),
                    as_float(item.get("discount_amount")),
                    line_total,
                    as_str(item.get("tax_code")),
                    as_str(
                        first_present(
                            item.get("vat_rate"), item.get("tax_rate"), item.get("tax_rate_percent")
                        )
                    ),
                    as_float(
                        first_present(item.get("confidence"), item.get("category_confidence"))
                    ),
                    as_str(
                        first_present(
                            item.get("review_status"),
                            receipt.get("human_review", {}).get("status")
                            if isinstance(receipt.get("human_review"), dict)
                            else None,
                        )
                    ),
                    embedding_text,
                    json.dumps(item, ensure_ascii=False, default=str),
                ),
            )
            item_id = int(cursor.lastrowid)
            if has_fts:
                connection.execute(
                    """
                    INSERT INTO receipt_item_fts(
                        item_id, receipt_id, merchant, receipt_date, raw_name,
                        normalized_name, category, embedding_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item_id,
                        receipt_id,
                        " ".join(value for value in [merchant_name, merchant_normalized] if value),
                        receipt_date,
                        description,
                        normalized_name,
                        category,
                        embedding_text,
                    ),
                )
            inserted += 1
        return inserted
