"""Canonical semantic documents built from structured receipt-item rows."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import Any

from receipt_intelligence.rag.models import ItemEmbeddingDocument

_WHITESPACE_RE = re.compile(r"\s+")
_PURCHASE_ITEM_TYPES = {
    "item",
    "product",
    "purchase_item",
    "purchased_product",
}

# Increment whenever the canonical text or its field precedence changes. Including
# the policy in the hash forces one safe incremental refresh after an upgrade.
ITEM_EMBEDDING_POLICY_VERSION = "approved_product_semantics_v3"

_PLACEHOLDER_DESCRIPTIONS = {
    "none",
    "null",
    "unknown",
    "n/a",
    "na",
    "item",
    "product",
    "product purchase",
    "purchase",
}


class UnindexableItemDescriptionError(ValueError):
    """Raised when a receipt row has no useful semantic product text."""


def clean_semantic_text(value: Any) -> str:
    """Normalize whitespace without translating or completing OCR text."""

    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def is_indexable_description(value: Any) -> bool:
    """Return whether a product description carries useful semantic content.

    This is conservative data-quality filtering, not product-specific business
    logic. Null-like values and generic placeholders should never consume an
    embedding or be presented to the LLM candidate resolver.
    """

    cleaned = clean_semantic_text(value)
    normalized = cleaned.casefold()
    return bool(cleaned) and len(normalized) >= 3 and normalized not in _PLACEHOLDER_DESCRIPTIONS


def build_item_embedding_document(row: Mapping[str, Any]) -> ItemEmbeddingDocument:
    """Create a deterministic semantic document from one SQL item row.

    Product identity and reviewed semantics are embedded. Prices, dates,
    quantities, merchants, and totals remain structured SQL values.
    """

    item_id = _required_positive_int(_first_present(row, "item_id", "id"), "item_id")
    receipt_id = _optional_positive_int(_first_present(row, "receipt_id"), "receipt_id")

    description = clean_semantic_text(
        _first_present(
            row,
            "description",
            "raw_name",
            "product_description",
            "name",
        )
    )
    if not is_indexable_description(description):
        raise UnindexableItemDescriptionError(
            f"Item {item_id} has no indexable product description: {description!r}."
        )

    normalized_description = clean_semantic_text(
        _first_present(
            row,
            "description_normalized",
            "normalized_description",
            "normalized_name",
        )
    )
    category = _reviewed_category_from_row(row)
    semantic_description = _semantic_description_from_row(row)
    merchant = clean_semantic_text(
        _first_present(row, "merchant", "merchant_name", "merchant_normalized")
    )
    parser_item_type = clean_semantic_text(
        _first_present(row, "parser_item_type", "line_type", "row_type")
    ).lower()

    document_type = (
        "purchased product"
        if parser_item_type in _PURCHASE_ITEM_TYPES
        else f"receipt line ({parser_item_type})"
        if parser_item_type
        else "receipt item"
    )

    lines = [
        f"Document type: {document_type}",
        f"Product description: {description}",
    ]

    if normalized_description and normalized_description.casefold() != description.casefold():
        lines.append(f"Normalized product description: {normalized_description}")
    if category:
        lines.append(f"Reviewed category: {category}")
    if semantic_description:
        lines.append(f"Reviewed semantic description: {semantic_description}")

    text = "\n".join(lines)
    hash_source = f"{ITEM_EMBEDDING_POLICY_VERSION}\n{text}"
    content_hash = sha256(hash_source.encode("utf-8")).hexdigest()

    return ItemEmbeddingDocument(
        item_id=item_id,
        receipt_id=receipt_id,
        description=description,
        normalized_description=normalized_description or None,
        category=category or None,
        semantic_description=semantic_description or None,
        merchant=merchant or None,
        parser_item_type=parser_item_type or None,
        text=text,
        content_hash=content_hash,
    )


def build_item_embedding_documents(
    rows: Iterable[Mapping[str, Any]],
) -> list[ItemEmbeddingDocument]:
    """Build documents in source-row order."""

    return [build_item_embedding_document(row) for row in rows]


def _raw_item_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _first_present(row, "item_raw_json", "raw_json")
    if isinstance(raw, Mapping):
        return raw
    if raw in (None, ""):
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _category_from_mapping(value: Mapping[str, Any]) -> str:
    path = clean_semantic_text(
        _first_present(
            value,
            "category_path",
            "product_category",
            "spending_category",
            "analytics_category",
        )
    )
    group = clean_semantic_text(value.get("category_group"))
    key = clean_semantic_text(value.get("category_key"))
    category = clean_semantic_text(value.get("category"))

    if path:
        return path[:500]
    if group and key:
        return f"{group} / {key}"[:500]
    if category:
        return category[:500]
    if key:
        return key[:500]
    if group:
        return group[:500]
    return ""


def _reviewed_category_from_row(row: Mapping[str, Any]) -> str:
    """Return the authoritative persisted category, then legacy JSON fallback."""

    direct = _category_from_mapping(row)
    if direct:
        return direct
    return _category_from_mapping(_raw_item_payload(row))


def _semantic_description_from_row(row: Mapping[str, Any]) -> str:
    """Return reviewed semantics from columns, then approved item JSON fallback."""

    direct = clean_semantic_text(_first_present(row, "semantic_description", "category_reason"))
    if direct:
        return direct[:2000]

    payload = _raw_item_payload(row)
    value = clean_semantic_text(_first_present(payload, "semantic_description", "category_reason"))
    return value[:2000]


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _required_positive_int(value: Any, field_name: str) -> int:
    parsed = _optional_positive_int(value, field_name)
    if parsed is None:
        raise ValueError(f"{field_name} is required.")
    return parsed


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return parsed


__all__ = [
    "ITEM_EMBEDDING_POLICY_VERSION",
    "UnindexableItemDescriptionError",
    "build_item_embedding_document",
    "build_item_embedding_documents",
    "clean_semantic_text",
    "is_indexable_description",
]
