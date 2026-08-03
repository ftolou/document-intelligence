"""Approved receipt persistence and lifecycle operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from receipt_intelligence.receipt_compat import (
    apply_review_field,
    apply_review_item_field,
    is_next_receipt,
    item_line_total,
    receipt_currency,
    receipt_grand_total,
    receipt_paid_total,
    receipt_payment_method,
    receipt_subtotal,
    receipt_tax_total,
)
from receipt_intelligence.receipt_compat import (
    receipt_date as compat_receipt_date,
)
from receipt_intelligence.receipt_compat import (
    receipt_time as compat_receipt_time,
)
from receipt_intelligence.storage.fingerprints import file_sha256, receipt_core
from receipt_intelligence.storage.models import ReceiptImportResult
from receipt_intelligence.storage.normalization import (
    as_float,
    as_str,
    build_item_embedding_text,
    category_from_item,
    extract_item_description,
    first_present,
    normalize_merchant_name,
    parser_item_type_from_item,
    utc_now,
)
from receipt_intelligence.storage.repositories.base import BaseRepository, fts_available
from receipt_intelligence.storage.repositories.items import ItemRepository


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    if value in (None, ""):
        return {}
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("stored JSON value must be an object")
    return parsed


def _without_internal_item_ids(receipt: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(receipt, ensure_ascii=False, default=str))
    items = cleaned.get("items") if isinstance(cleaned.get("items"), list) else []
    for item in items:
        if isinstance(item, dict):
            item.pop("_db_item_id", None)
    return cleaned


def _semantic_signature(
    *,
    description: str | None,
    normalized_name: str | None,
    parser_item_type: str | None,
    category: str | None,
    semantic_description: str | None,
) -> tuple[str, str, str, str, str]:
    """Return exactly the fields used by embedding policy v3."""

    def normalized(value: Any) -> str:
        return " ".join(str(value or "").split()).casefold()

    return (
        normalized(description),
        normalized(normalized_name),
        normalized(parser_item_type),
        normalized(category),
        normalized(semantic_description),
    )


def _semantic_description(item: dict[str, Any]) -> str | None:
    return as_str(first_present(item.get("semantic_description"), item.get("category_reason")))


def _category_path(group: Any, key: Any) -> str | None:
    group_text = as_str(group)
    key_text = as_str(key)
    if group_text and key_text:
        return f"{group_text}/{key_text}"
    return group_text or key_text


def _review_category(item: dict[str, Any], description: str) -> str | None:
    """Prefer explicit human category fields over automatic aliases."""

    explicit = first_present(
        item.get("category_path"),
        _category_path(item.get("category_group"), item.get("category_key")),
        item.get("product_category"),
        item.get("spending_category"),
        item.get("analytics_category"),
    )
    return as_str(explicit) or category_from_item(item, description)


class ReceiptRepository(BaseRepository):
    def __init__(self, connections, items: ItemRepository) -> None:
        super().__init__(connections)
        self.items = items

    def import_receipt(
        self,
        *,
        job_id: str,
        receipt: dict[str, Any],
        approved_receipt_path: Path | None = None,
        source_receipt_path: Path | None = None,
        image_path: Path | str | None = None,
    ) -> ReceiptImportResult:
        merchant = receipt.get("merchant") if isinstance(receipt.get("merchant"), dict) else {}
        human_review = (
            receipt.get("human_review") if isinstance(receipt.get("human_review"), dict) else {}
        )
        validation = (
            receipt.get("validation") if isinstance(receipt.get("validation"), dict) else {}
        )
        payment_method = as_str(receipt_payment_method(receipt))

        merchant_name = as_str(first_present(merchant.get("name"), receipt.get("merchant_name")))
        merchant_normalized = normalize_merchant_name(merchant_name)
        now = utc_now()
        raw_json = json.dumps(receipt, ensure_ascii=False, default=str)
        core = receipt_core(receipt)
        image_hash = file_sha256(image_path)

        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM receipts WHERE job_id = ?", (job_id,)
            ).fetchone()
            values = (
                merchant_name,
                merchant_normalized,
                as_str(compat_receipt_date(receipt)),
                as_str(compat_receipt_time(receipt)),
                as_str(receipt_currency(receipt)) or "EUR",
                as_float(receipt_subtotal(receipt)),
                as_float(receipt_tax_total(receipt)),
                as_float(receipt_grand_total(receipt)),
                as_float(receipt_paid_total(receipt)),
                payment_method,
                as_str(
                    first_present(
                        human_review.get("status"),
                        validation.get("import_decision"),
                        receipt.get("parse_status"),
                    )
                ),
                as_str(human_review.get("reviewer")),
                str(image_path) if image_path else None,
                str(approved_receipt_path) if approved_receipt_path else None,
                str(source_receipt_path) if source_receipt_path else None,
                raw_json,
                image_hash,
                core.get("content_fingerprint"),
            )
            if existing:
                receipt_id = int(existing["id"])
                if fts_available(connection):
                    connection.execute(
                        "DELETE FROM receipt_item_fts WHERE receipt_id = ?",
                        (receipt_id,),
                    )
                connection.execute("DELETE FROM receipt_items WHERE receipt_id = ?", (receipt_id,))
                connection.execute(
                    """
                    UPDATE receipts SET
                        merchant_name=?, merchant_normalized=?, receipt_date=?,
                        receipt_time=?, currency=?, subtotal=?, tax_total=?,
                        grand_total=?, paid_total=?, payment_method=?, review_status=?,
                        reviewer=?, image_path=?, approved_receipt_path=?,
                        source_receipt_path=?, raw_json=?, file_sha256=?,
                        content_fingerprint=?, updated_at=?
                    WHERE id=?
                    """,
                    (*values, now, receipt_id),
                )
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO receipts(
                        job_id, merchant_name, merchant_normalized, receipt_date,
                        receipt_time, currency, subtotal, tax_total, grand_total,
                        paid_total, payment_method, review_status, reviewer, image_path,
                        approved_receipt_path, source_receipt_path, raw_json, file_sha256,
                        content_fingerprint, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, *values, now, now),
                )
                receipt_id = int(cursor.lastrowid)

            item_count = self.items.insert_items(
                connection,
                receipt_id,
                receipt,
                merchant_name,
                merchant_normalized,
            )
            connection.commit()
        return ReceiptImportResult(
            receipt_db_id=receipt_id,
            job_id=job_id,
            item_count=item_count,
            inserted_at=now,
        )

    def list_receipts(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, job_id, merchant_name, merchant_normalized,
                       receipt_date, receipt_time, grand_total, currency,
                       review_status, item_count, approved_receipt_path,
                       image_path, updated_at
                FROM (
                    SELECT r.*, COUNT(i.id) AS item_count
                    FROM receipts r
                    LEFT JOIN receipt_items i ON i.receipt_id = r.id
                    GROUP BY r.id
                )
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, min(1000, int(limit))),),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_receipt(self, receipt_id: int) -> dict[str, Any] | None:
        """Return safe receipt metadata used for in-app navigation."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT r.id, r.job_id, r.merchant_name, r.merchant_normalized,
                       r.receipt_date, r.receipt_time, r.grand_total, r.currency,
                       r.review_status, r.updated_at, COUNT(i.id) AS item_count
                FROM receipts AS r
                LEFT JOIN receipt_items AS i ON i.receipt_id = r.id
                WHERE r.id = ?
                GROUP BY r.id
                """,
                (int(receipt_id),),
            ).fetchone()
            return dict(row) if row is not None else None

    def get_receipt_edit_document(self, receipt_id: int) -> dict[str, Any] | None:
        """Reconstruct the editable receipt from authoritative relational rows."""

        with self.connect() as connection:
            receipt_row = connection.execute(
                """
                SELECT id, job_id, merchant_name, merchant_normalized, receipt_date,
                       receipt_time, currency, subtotal, tax_total, grand_total, paid_total,
                       payment_method, review_status, reviewer, image_path,
                       approved_receipt_path, source_receipt_path, raw_json, updated_at
                FROM receipts
                WHERE id = ?
                """,
                (int(receipt_id),),
            ).fetchone()
            if receipt_row is None:
                return None
            item_rows = connection.execute(
                """
                SELECT id, item_index, raw_name, normalized_name, category,
                       parser_item_type, category_group, category_key, category_reason,
                       semantic_description, quantity, unit, unit_price, original_price,
                       discount_amount, line_total, tax_code,
                       vat_rate, confidence, review_status, raw_json
                FROM receipt_items
                WHERE receipt_id = ?
                ORDER BY item_index, id
                """,
                (int(receipt_id),),
            ).fetchall()

        row = dict(receipt_row)
        receipt = _json_object(row.get("raw_json"))
        merchant = receipt.get("merchant") if isinstance(receipt.get("merchant"), dict) else {}
        merchant = dict(merchant)
        merchant["name"] = row.get("merchant_name")
        receipt["merchant"] = merchant
        apply_review_field(receipt, "date", row.get("receipt_date"))
        apply_review_field(receipt, "time", row.get("receipt_time"))
        apply_review_field(receipt, "currency", row.get("currency") or "EUR")
        apply_review_field(receipt, "subtotal", row.get("subtotal"))
        apply_review_field(receipt, "tax_total", row.get("tax_total"))
        apply_review_field(receipt, "grand_total", row.get("grand_total"))
        apply_review_field(receipt, "paid_total", row.get("paid_total"))
        apply_review_field(receipt, "payment_method", row.get("payment_method"))

        review = (
            receipt.get("human_review") if isinstance(receipt.get("human_review"), dict) else {}
        )
        review = dict(review)
        review["status"] = row.get("review_status") or review.get("status")
        review["reviewer"] = row.get("reviewer") or review.get("reviewer")
        receipt["human_review"] = review

        items: list[dict[str, Any]] = []
        for stored in item_rows:
            item_row = dict(stored)
            item = _json_object(item_row.get("raw_json"))
            item["_db_item_id"] = int(item_row["id"])
            next_item_schema = is_next_receipt({"items": [item]})
            if next_item_schema:
                item["name"] = item_row.get("raw_name")
            else:
                item["product_description"] = item_row.get("raw_name")
                item.setdefault("description", item_row.get("raw_name"))
            item["normalized_name"] = item_row.get("normalized_name")
            item["parser_item_type"] = item_row.get("parser_item_type")
            if not next_item_schema:
                item["receipt_row_type"] = item_row.get("parser_item_type")
            item["category_group"] = item_row.get("category_group")
            item["category_key"] = item_row.get("category_key")
            item["category_reason"] = item_row.get("category_reason")
            item["semantic_description"] = item_row.get("semantic_description")
            category_path = _category_path(
                item_row.get("category_group"), item_row.get("category_key")
            )
            if category_path:
                item["category_path"] = category_path
            if item_row.get("category"):
                item["product_category"] = item_row.get("category")
            if item_row.get("parser_item_type") and not next_item_schema:
                item["category"] = item_row.get("parser_item_type")
            for field in (
                "quantity",
                "unit",
                "unit_price",
                "original_price",
                "discount_amount",
                "line_total",
                "tax_code",
                "vat_rate",
                "confidence",
                "review_status",
            ):
                apply_review_item_field(
                    item,
                    field,
                    item_row.get(field),
                    next_schema=next_item_schema,
                )
            items.append(item)
        receipt["items"] = items
        receipt["_database"] = {
            "receipt_id": int(row["id"]),
            "job_id": row.get("job_id"),
            "updated_at": row.get("updated_at"),
        }
        return receipt

    def list_receipt_item_ids(self, receipt_id: int) -> list[int]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM receipt_items WHERE receipt_id = ? ORDER BY item_index, id",
                (int(receipt_id),),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def update_receipt_from_review(
        self,
        receipt_id: int,
        receipt: dict[str, Any],
        *,
        expected_job_id: str,
        expected_updated_at: str,
    ) -> dict[str, Any]:
        """Persist a reviewed receipt transactionally without replacing item identities.

        Semantic embeddings are invalidated only for rows whose canonical product
        document changed. Lexical/FTS metadata is refreshed for the complete receipt.
        """

        if not isinstance(receipt, dict):
            raise ValueError("receipt must be an object")
        incoming_items = receipt.get("items")
        if not isinstance(incoming_items, list):
            raise ValueError("receipt items must be a list")

        merchant = receipt.get("merchant") if isinstance(receipt.get("merchant"), dict) else {}
        human_review = (
            receipt.get("human_review") if isinstance(receipt.get("human_review"), dict) else {}
        )
        validation = (
            receipt.get("validation") if isinstance(receipt.get("validation"), dict) else {}
        )
        merchant_name = as_str(first_present(merchant.get("name"), receipt.get("merchant_name")))
        merchant_normalized = normalize_merchant_name(merchant_name)
        receipt_date = as_str(compat_receipt_date(receipt))
        currency = as_str(receipt_currency(receipt)) or "EUR"
        now = utc_now()

        with self.connect() as connection:
            stored_receipt = connection.execute(
                "SELECT id, job_id, review_status, updated_at FROM receipts WHERE id = ?",
                (int(receipt_id),),
            ).fetchone()
            if stored_receipt is None:
                raise KeyError("receipt not found")
            if as_str(stored_receipt["job_id"]) != as_str(expected_job_id):
                raise ValueError("review job identity changed; reload the receipt")
            if as_str(stored_receipt["updated_at"]) != as_str(expected_updated_at):
                raise ValueError("review state changed while saving; reload the receipt")
            stored_items = connection.execute(
                """
                SELECT id, item_index, raw_name, normalized_name, category,
                       parser_item_type, category_group, category_key, category_reason,
                       semantic_description, quantity, unit, unit_price, original_price,
                       discount_amount, line_total, tax_code,
                       vat_rate, confidence, review_status, raw_json
                FROM receipt_items
                WHERE receipt_id = ?
                ORDER BY item_index, id
                """,
                (int(receipt_id),),
            ).fetchall()
            stored_by_id = {int(row["id"]): dict(row) for row in stored_items}
            previous_review_status = as_str(stored_receipt["review_status"])

            if len(incoming_items) != len(stored_items):
                raise ValueError(
                    "The database editor currently requires the existing item "
                    "row count to remain unchanged."
                )

            resolved_items: list[tuple[int, int, dict[str, Any]]] = []
            seen_item_ids: set[int] = set()
            for index, item in enumerate(incoming_items):
                if not isinstance(item, dict):
                    raise ValueError(f"item {index} must be an object")
                raw_id = item.get("_db_item_id")
                if raw_id in (None, ""):
                    raise ValueError(f"item {index} is missing its database identity")
                item_id = int(raw_id)
                if item_id not in stored_by_id or item_id in seen_item_ids:
                    raise ValueError(f"item {index} has an invalid database identity")
                seen_item_ids.add(item_id)
                resolved_items.append((item_id, index, item))

            all_item_ids = sorted(stored_by_id)
            current_review_status = as_str(human_review.get("status")) or "needs_review"
            semantic_item_ids: list[int] = []
            metadata_item_ids: list[int] = []
            changed_item_ids: list[int] = []
            item_payloads: list[tuple[int, int, dict[str, Any], tuple[Any, ...]]] = []

            for item_id, index, item in resolved_items:
                stored = stored_by_id[item_id]
                description = extract_item_description(item).strip()
                if not description:
                    raise ValueError(f"item {index + 1} requires a product description")
                normalized_name = as_str(
                    first_present(
                        item.get("normalized_name"),
                        item.get("product_description"),
                        item.get("name"),
                        description,
                    )
                )
                category = _review_category(item, description)
                parser_item_type = parser_item_type_from_item(item)
                line_total = as_float(item_line_total(item))
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
                clean_item = json.loads(json.dumps(item, ensure_ascii=False, default=str))
                clean_item.pop("_db_item_id", None)
                values = (
                    index,
                    description,
                    normalized_name,
                    category,
                    parser_item_type,
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
                            item.get("vat_rate"),
                            item.get("tax_rate"),
                            item.get("tax_rate_percent"),
                        )
                    ),
                    as_float(
                        first_present(item.get("confidence"), item.get("category_confidence"))
                    ),
                    as_str(first_present(item.get("review_status"), human_review.get("status"))),
                    embedding_text,
                    json.dumps(clean_item, ensure_ascii=False, default=str),
                )
                stored_raw_item = _json_object(stored.get("raw_json"))
                stored_semantic_item = dict(stored_raw_item)
                for key in (
                    "category",
                    "category_group",
                    "category_key",
                    "category_reason",
                    "semantic_description",
                ):
                    value = stored.get(key)
                    if value not in (None, ""):
                        stored_semantic_item[key] = value

                old_hash = _semantic_signature(
                    description=stored.get("raw_name"),
                    normalized_name=stored.get("normalized_name"),
                    parser_item_type=stored.get("parser_item_type"),
                    category=_review_category(
                        stored_semantic_item, str(stored.get("raw_name") or "")
                    ),
                    semantic_description=_semantic_description(stored_semantic_item),
                )
                new_hash = _semantic_signature(
                    description=description,
                    normalized_name=normalized_name,
                    parser_item_type=parser_item_type,
                    category=category,
                    semantic_description=_semantic_description(item),
                )
                stored_values = (
                    stored.get("item_index"),
                    stored.get("raw_name"),
                    stored.get("normalized_name"),
                    stored.get("category"),
                    stored.get("parser_item_type"),
                    stored.get("category_group"),
                    stored.get("category_key"),
                    stored.get("category_reason"),
                    stored.get("semantic_description"),
                    stored.get("quantity"),
                    stored.get("unit"),
                    stored.get("unit_price"),
                    stored.get("original_price"),
                    stored.get("discount_amount"),
                    stored.get("line_total"),
                    stored.get("tax_code"),
                    stored.get("vat_rate"),
                    stored.get("confidence"),
                    stored.get("review_status"),
                    stored.get("embedding_text"),
                    stored.get("raw_json"),
                )
                if values != stored_values:
                    changed_item_ids.append(item_id)
                    if old_hash != new_hash:
                        semantic_item_ids.append(item_id)
                    else:
                        metadata_item_ids.append(item_id)
                item_payloads.append((item_id, index, clean_item, values))

            clean_receipt = _without_internal_item_ids(receipt)
            clean_receipt.pop("_database", None)
            clean_receipt["items"] = [payload[2] for payload in item_payloads]
            core = receipt_core(clean_receipt)
            payment_method = as_str(receipt_payment_method(clean_receipt))

            connection.execute(
                """
                UPDATE receipts SET
                    merchant_name=?, merchant_normalized=?, receipt_date=?,
                    receipt_time=?, currency=?, subtotal=?, tax_total=?,
                    grand_total=?, paid_total=?, payment_method=?, review_status=?,
                    reviewer=?, raw_json=?, content_fingerprint=?, updated_at=?
                WHERE id=?
                """,
                (
                    merchant_name,
                    merchant_normalized,
                    receipt_date,
                    as_str(compat_receipt_time(clean_receipt)),
                    currency,
                    as_float(receipt_subtotal(clean_receipt)),
                    as_float(receipt_tax_total(clean_receipt)),
                    as_float(receipt_grand_total(clean_receipt)),
                    as_float(receipt_paid_total(clean_receipt)),
                    payment_method,
                    as_str(
                        first_present(
                            human_review.get("status"),
                            validation.get("import_decision"),
                            clean_receipt.get("parse_status"),
                        )
                    ),
                    as_str(human_review.get("reviewer")),
                    json.dumps(clean_receipt, ensure_ascii=False, default=str),
                    core.get("content_fingerprint"),
                    now,
                    int(receipt_id),
                ),
            )

            for item_id, _index, _clean_item, values in item_payloads:
                connection.execute(
                    """
                    UPDATE receipt_items SET
                        item_index=?, raw_name=?, normalized_name=?, category=?,
                        parser_item_type=?, category_group=?, category_key=?, category_reason=?,
                        semantic_description=?, quantity=?, unit=?, unit_price=?, original_price=?,
                        discount_amount=?,
                        line_total=?, tax_code=?, vat_rate=?, confidence=?,
                        review_status=?, embedding_text=?, raw_json=?
                    WHERE id=? AND receipt_id=?
                    """,
                    (*values, item_id, int(receipt_id)),
                )

            if fts_available(connection):
                connection.execute(
                    "DELETE FROM receipt_item_fts WHERE receipt_id = ?",
                    (int(receipt_id),),
                )
                connection.execute(
                    """
                    INSERT INTO receipt_item_fts(
                        item_id, receipt_id, merchant, receipt_date, raw_name,
                        normalized_name, category, embedding_text
                    )
                    SELECT i.id, i.receipt_id,
                           trim(COALESCE(r.merchant_name, '') || ' ' ||
                                COALESCE(r.merchant_normalized, '')),
                           r.receipt_date, i.raw_name, i.normalized_name,
                           i.category, i.embedding_text
                    FROM receipt_items AS i
                    JOIN receipts AS r ON r.id = i.receipt_id
                    WHERE i.receipt_id = ?
                    """,
                    (int(receipt_id),),
                )

            invalidated_embedding_count = 0
            embedding_invalidation_ids = (
                all_item_ids if current_review_status != "approved" else semantic_item_ids
            )
            if embedding_invalidation_ids:
                placeholders = ", ".join("?" for _ in embedding_invalidation_ids)
                cursor = connection.execute(
                    f"DELETE FROM rag_item_embeddings WHERE item_id IN ({placeholders})",
                    embedding_invalidation_ids,
                )
                invalidated_embedding_count = max(0, int(cursor.rowcount or 0))

            job_id = stored_receipt["job_id"]
            if job_id:
                connection.execute(
                    """
                    UPDATE review_queue
                    SET queue_status=?, receipt_db_id=?, decision=?, balanced=?,
                        difference=?, issue_count=?, raw_json=?, updated_at=?
                    WHERE job_id=?
                    """,
                    (
                        current_review_status,
                        int(receipt_id),
                        as_str(validation.get("import_decision")),
                        (
                            1
                            if validation.get("balanced") is True
                            else 0
                            if validation.get("balanced") is False
                            else None
                        ),
                        as_float(validation.get("difference")),
                        len(validation.get("issues") or [])
                        if isinstance(validation.get("issues"), list)
                        else 0,
                        json.dumps(clean_receipt, ensure_ascii=False, default=str),
                        now,
                        job_id,
                    ),
                )
            connection.commit()

        return {
            "receipt_db_id": int(receipt_id),
            "job_id": stored_receipt["job_id"],
            "item_count": len(item_payloads),
            "changed_item_ids": sorted(changed_item_ids),
            "all_item_ids": all_item_ids,
            "semantic_item_ids": sorted(semantic_item_ids),
            "metadata_item_ids": sorted(metadata_item_ids),
            "previous_review_status": previous_review_status,
            "review_status": current_review_status,
            "invalidated_embedding_count": invalidated_embedding_count,
            "updated_at": now,
        }

    def get_receipt_review_record(self, receipt_id: int) -> dict[str, Any] | None:
        """Return the internal persistence record required for durable review loading."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, job_id, merchant_name, merchant_normalized, receipt_date,
                       receipt_time, grand_total, currency, review_status, reviewer,
                       image_path, approved_receipt_path, source_receipt_path,
                       raw_json, updated_at
                FROM receipts
                WHERE id = ?
                """,
                (int(receipt_id),),
            ).fetchone()
            return dict(row) if row is not None else None

    def get_receipt_review_record_by_job_id(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        """Return the internal persistence record for a linked processing job."""

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, job_id, merchant_name, merchant_normalized, receipt_date,
                       receipt_time, grand_total, currency, review_status, reviewer,
                       image_path, approved_receipt_path, source_receipt_path,
                       raw_json, updated_at
                FROM receipts
                WHERE job_id = ?
                """,
                (str(job_id),),
            ).fetchone()
            return dict(row) if row is not None else None

    def delete_receipt(
        self,
        receipt_id: int | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        if receipt_id is None and not job_id:
            raise ValueError("receipt_id or job_id is required")
        with self.connect() as connection:
            if receipt_id is not None:
                row = connection.execute(
                    "SELECT id, job_id FROM receipts WHERE id=?", (receipt_id,)
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT id, job_id FROM receipts WHERE job_id=?", (job_id,)
                ).fetchone()
            if not row:
                return {"deleted": False, "reason": "not_found"}
            database_id = int(row["id"])
            stored_job_id = row["job_id"]
            if fts_available(connection):
                connection.execute(
                    "DELETE FROM receipt_item_fts WHERE receipt_id=?", (database_id,)
                )
            connection.execute("DELETE FROM receipt_items WHERE receipt_id=?", (database_id,))
            connection.execute("DELETE FROM receipts WHERE id=?", (database_id,))
            if stored_job_id:
                connection.execute(
                    """
                    UPDATE review_queue
                    SET receipt_db_id=NULL, queue_status='needs_review', updated_at=?
                    WHERE job_id=?
                    """,
                    (utc_now(), stored_job_id),
                )
            connection.commit()
        return {
            "deleted": True,
            "receipt_db_id": database_id,
            "job_id": stored_job_id,
        }

    def delete_all_receipt_data(self, *, include_review_queue: bool = False) -> dict[str, Any]:
        with self.connect() as connection:
            counts = {
                "receipts": int(
                    connection.execute("SELECT COUNT(*) AS n FROM receipts").fetchone()["n"]
                ),
                "items": int(
                    connection.execute("SELECT COUNT(*) AS n FROM receipt_items").fetchone()["n"]
                ),
                "review_queue": int(
                    connection.execute("SELECT COUNT(*) AS n FROM review_queue").fetchone()["n"]
                ),
                "review_history": int(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM receipt_review_history"
                    ).fetchone()["n"]
                ),
            }
            if fts_available(connection):
                connection.execute("DELETE FROM receipt_item_fts")
            connection.execute("DELETE FROM receipt_items")
            connection.execute("DELETE FROM receipts")
            connection.execute("DELETE FROM duplicate_candidates")
            if include_review_queue:
                connection.execute("DELETE FROM receipt_review_history")
                connection.execute("DELETE FROM review_queue")
            else:
                connection.execute(
                    """
                    UPDATE review_queue
                    SET receipt_db_id=NULL, queue_status='needs_review', updated_at=?
                    """,
                    (utc_now(),),
                )
            connection.commit()
        counts["include_review_queue"] = include_review_queue
        return counts
