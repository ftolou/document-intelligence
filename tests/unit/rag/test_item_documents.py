"""Unit tests for deterministic item embedding documents."""

from __future__ import annotations

import pytest

from receipt_intelligence.rag.item_documents import (
    build_item_embedding_document,
    build_item_embedding_documents,
)


def test_builds_reviewed_semantic_document_from_sql_row() -> None:
    document = build_item_embedding_document(
        {
            "item_id": 42,
            "receipt_id": 9,
            "raw_name": "  DAMEN   SNEAKER ",
            "normalized_name": "damen sneaker",
            "category_key": "clothing/shoes",
            "merchant_name": "SB-Warenhaus GmbH",
            "parser_item_type": "item",
            "line_total": 29.99,
            "receipt_date": "2026-06-20",
        }
    )

    assert document.item_id == 42
    assert document.receipt_id == 9
    assert document.text == (
        "Document type: purchased product\n"
        "Product description: DAMEN SNEAKER\n"
        "Reviewed category: clothing/shoes"
    )
    assert document.category == "clothing/shoes"
    assert document.merchant == "SB-Warenhaus GmbH"
    assert "29.99" not in document.text
    assert "2026-06-20" not in document.text
    assert "Reviewed category: clothing/shoes" in document.text
    assert "Warenhaus" not in document.text
    assert len(document.content_hash) == 64


def test_reviewed_category_and_reason_change_semantic_hash() -> None:
    base = {
        "item_id": 7,
        "raw_name": "VITTEL",
        "normalized_name": "vittel",
        "parser_item_type": "item",
    }

    water = build_item_embedding_document(
        {
            **base,
            "category": "beverages",
            "item_raw_json": '{"category_reason":"Vittel is a brand of mineral water."}',
        }
    )
    sparkling = build_item_embedding_document(
        {
            **base,
            "category": "beverages/water",
            "item_raw_json": '{"category_reason":"Still mineral water, now sparkling."}',
        }
    )

    assert "Reviewed category: beverages" in water.text
    assert "Reviewed semantic description: Vittel is a brand of mineral water." in water.text
    assert water.semantic_description == "Vittel is a brand of mineral water."
    assert water.content_hash != sparkling.content_hash


def test_persisted_review_fields_override_stale_raw_json() -> None:
    document = build_item_embedding_document(
        {
            "item_id": 8,
            "raw_name": "VITTEL",
            "category": "beverages/water",
            "category_reason": "Reviewed current reason.",
            "item_raw_json": ('{"category":"unknown","category_reason":"Stale reason."}'),
        }
    )

    assert "Reviewed category: beverages/water" in document.text
    assert "Reviewed semantic description: Reviewed current reason." in document.text
    assert "unknown" not in document.text
    assert "Stale reason" not in document.text


def test_non_semantic_metadata_changes_do_not_change_hash() -> None:
    base = {
        "item_id": 7,
        "raw_name": "VITTEL",
        "category": "beverages/water",
        "category_reason": "Vittel is a brand of mineral water.",
    }
    first = build_item_embedding_document(
        {**base, "merchant": "LIDL", "line_total": 5.10, "receipt_date": "2026-06-01"}
    )
    second = build_item_embedding_document(
        {**base, "merchant": "OTHER", "line_total": 6.00, "receipt_date": "2026-07-01"}
    )

    assert first.text == second.text
    assert first.content_hash == second.content_hash


def test_product_name_change_updates_hash() -> None:
    first = build_item_embedding_document(
        {
            "item_id": 7,
            "raw_name": "MINERALWASSER",
        }
    )
    second = build_item_embedding_document(
        {
            "item_id": 7,
            "raw_name": "MINERALWASSER CLASSIC",
        }
    )

    assert first.content_hash != second.content_hash


def test_does_not_expand_or_invent_truncated_product_words() -> None:
    document = build_item_embedding_document(
        {
            "item_id": 11,
            "raw_name": "BIO TK SCHNITTLA",
            "normalized_name": "BIO TK SCHNITTLA",
            "parser_item_type": "item",
        }
    )

    assert "BIO TK SCHNITTLA" in document.text
    assert "SCHNITTLAUCH" not in document.text


def test_build_many_preserves_source_order() -> None:
    documents = build_item_embedding_documents(
        [
            {"item_id": 2, "raw_name": "SECOND"},
            {"item_id": 1, "raw_name": "FIRST"},
        ]
    )

    assert [document.item_id for document in documents] == [2, 1]


def test_requires_positive_item_id_and_description() -> None:
    with pytest.raises(ValueError, match="item_id"):
        build_item_embedding_document({"raw_name": "WASSER"})
    with pytest.raises(ValueError, match="description"):
        build_item_embedding_document({"item_id": 1})


def test_rejects_null_like_and_generic_placeholder_descriptions() -> None:
    for description in ("None", "unknown", "Product Purchase", "item"):
        with pytest.raises(ValueError, match="no indexable product description"):
            build_item_embedding_document(
                {
                    "item_id": 99,
                    "raw_name": description,
                    "parser_item_type": "item",
                }
            )


def test_embedding_document_includes_reviewed_semantic_description() -> None:
    document = build_item_embedding_document(
        {
            "item_id": 42,
            "receipt_id": 7,
            "description": "VITTEL 1,5L",
            "normalized_name": "Vittel",
            "semantic_description": "Vittel is bottled mineral water.",
            "parser_item_type": "item",
            "merchant": "REWE",
        }
    )

    assert "Reviewed semantic description: Vittel is bottled mineral water." in document.text
    assert "REWE" not in document.text
