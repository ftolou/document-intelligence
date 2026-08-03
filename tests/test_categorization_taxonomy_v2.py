from __future__ import annotations

from receipt_intelligence.extraction.categorization.items import (
    CATEGORY_SCHEMA_VERSION,
    _coerce_categories,
    _coerce_merchant_classification,
    build_categorization_prompt,
)
from receipt_intelligence.domain.categorization_taxonomy import (
    ITEM_TAXONOMY_VERSION,
    MERCHANT_TAXONOMY_VERSION,
    canonical_item_category_key,
    fashion_category_path,
)
from receipt_intelligence.services.review_service import _category_path_from_group_key
from receipt_intelligence.storage.normalization import category_from_item


def _modepark_items() -> list[dict[str, object]]:
    return [
        {
            "description": "KRAWATTE",
            "product_description": "KRAWATTE",
            "original_price": 14.99,
            "discount_amount": 0.75,
            "line_total": 14.24,
        },
        {
            "description": "Louie Winter",
            "product_description": "Louie Winter",
            "original_price": 120.0,
            "discount_amount": 24.0,
            "line_total": 96.0,
        },
    ]


def test_prompt_uses_v2_taxonomy_and_formal_schema() -> None:
    receipt = {
        "merchant": {
            "name": "Modepark Röther",
            "address": {
                "street": "Josef-Landes-Straße 44",
                "postal_code": "87600",
                "city": "Kaufbeuren",
            },
        },
        "currency": "EUR",
        "items": _modepark_items(),
    }

    prompt = build_categorization_prompt(receipt)

    assert CATEGORY_SCHEMA_VERSION in prompt
    assert ITEM_TAXONOMY_VERSION in prompt
    assert MERCHANT_TAXONOMY_VERSION in prompt
    assert '"type": "object"' in prompt
    assert '"enum"' in prompt
    assert "fashion_accessories" in prompt
    assert "fashion_footwear" in prompt
    assert "fashion_unknown" in prompt
    assert "clothing_shoes|" not in prompt
    assert "KRAWATTE" in prompt


def test_modepark_tie_is_accessory_not_footwear() -> None:
    parsed = {
        "schema_version": CATEGORY_SCHEMA_VERSION,
        "taxonomy_version": ITEM_TAXONOMY_VERSION,
        "merchant_taxonomy_version": MERCHANT_TAXONOMY_VERSION,
        "merchant_classification": {
            "category_key": "fashion_retail",
            "confidence": 0.95,
            "reason": "Modepark Röther is a fashion retailer.",
        },
        "items": [
            {
                "item_index": 0,
                "category_key": "fashion_accessories",
                "confidence": 0.98,
                "text_certainty": "explicit",
                "evidence_terms": ["KRAWATTE"],
                "reason": "The printed item is explicitly a tie.",
            },
            {
                "item_index": 1,
                "category_key": "fashion_apparel",
                "confidence": 0.72,
                "text_certainty": "contextual",
                "evidence_terms": ["Louie Winter", "Modepark Röther"],
                "reason": "Fashion context is supported, but the subtype is not explicit.",
            },
        ],
        "warnings": [],
    }

    merchant, merchant_warnings = _coerce_merchant_classification(parsed)
    categories, category_warnings = _coerce_categories(
        parsed,
        _modepark_items(),
        merchant_context_text="Modepark Röther fashion_retail",
    )

    assert not merchant_warnings
    assert merchant["category_key"] == "fashion_retail"
    assert categories[0]["category_key"] == "fashion_accessories"
    assert categories[0]["category_path"] == "fashion/accessories"
    assert categories[0]["category_group"] == "Fashion"
    assert categories[1]["category_key"] == "fashion_unknown"
    assert categories[1]["category_path"] == "fashion/unknown"
    assert any("downgraded contextual fashion subtype" in warning.lower() for warning in category_warnings)
    assert all(row["category_key"] != "fashion_footwear" for row in categories)


def test_legacy_combined_key_never_becomes_footwear() -> None:
    assert canonical_item_category_key("clothing_shoes") == "fashion_unknown"
    assert fashion_category_path("clothing_shoes") == "fashion/unknown"

    categories, warnings = _coerce_categories(
        {
            "items": [
                {
                    "item_index": 0,
                    "category_key": "clothing_shoes",
                    "confidence": 0.9,
                    "text_certainty": "contextual",
                    "evidence_terms": ["KRAWATTE"],
                    "reason": "Legacy category output.",
                }
            ]
        },
        [_modepark_items()[0]],
        merchant_context_text="Modepark Röther",
    )

    assert categories[0]["category_key"] == "fashion_unknown"
    assert categories[0]["category_path"] == "fashion/unknown"
    assert categories[0]["category_key"] != "fashion_footwear"
    assert any("legacy" in warning.lower() for warning in warnings)


def test_review_save_preserves_canonical_fashion_path() -> None:
    assert (
        _category_path_from_group_key("Fashion", "fashion_accessories")
        == "fashion/accessories"
    )
    assert (
        _category_path_from_group_key("Clothing", "clothing_shoes")
        == "fashion/unknown"
    )


def test_storage_uses_canonical_fashion_paths() -> None:
    assert (
        category_from_item({"category_key": "fashion_accessories"}, "KRAWATTE")
        == "fashion/accessories"
    )
    assert (
        category_from_item({"category_key": "clothing_shoes"}, "KRAWATTE")
        == "fashion/unknown"
    )
