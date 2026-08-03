"""Versioned receipt categorization taxonomy helpers.

The v2 taxonomy separates merchant verticals from purchased-item semantics and,
most importantly, no longer uses one overloaded ``clothing_shoes`` category for
apparel, footwear, sportswear, and fashion accessories.
"""

from __future__ import annotations

from typing import Any

ITEM_TAXONOMY_VERSION = "receipt_item_taxonomy_v2"
MERCHANT_TAXONOMY_VERSION = "receipt_merchant_taxonomy_v2"

FASHION_ITEM_TAXONOMY: tuple[dict[str, str], ...] = (
    {
        "key": "fashion_apparel",
        "group": "Fashion",
        "path": "fashion/apparel",
        "description": (
            "General clothing such as shirts, trousers, dresses, jackets, and coats; "
            "not footwear, accessories, underwear, or explicit sportswear"
        ),
    },
    {
        "key": "fashion_footwear",
        "group": "Fashion",
        "path": "fashion/footwear",
        "description": "Shoes, boots, sandals, slippers, and other explicit footwear",
    },
    {
        "key": "fashion_accessories",
        "group": "Fashion",
        "path": "fashion/accessories",
        "description": "Ties, belts, scarves, hats, bags, wallets, and fashion accessories",
    },
    {
        "key": "fashion_sportswear",
        "group": "Fashion",
        "path": "fashion/sportswear",
        "description": "Explicit sports clothing and athletic wear; footwear remains fashion_footwear",
    },
    {
        "key": "fashion_underwear",
        "group": "Fashion",
        "path": "fashion/underwear",
        "description": "Underwear, lingerie, socks, hosiery, and base layers",
    },
    {
        "key": "fashion_unknown",
        "group": "Fashion",
        "path": "fashion/unknown",
        "description": (
            "Fashion item supported by merchant or peer context, but the printed item text "
            "does not justify a more specific fashion subtype"
        ),
    },
)

SPECIFIC_FASHION_ITEM_KEYS = frozenset(
    row["key"] for row in FASHION_ITEM_TAXONOMY if row["key"] != "fashion_unknown"
)

FASHION_MERCHANT_TAXONOMY: tuple[dict[str, str], ...] = (
    {
        "key": "fashion_retail",
        "description": "Clothing, fashion, department-fashion, or mixed apparel retailer",
    },
    {
        "key": "footwear_retail",
        "description": "Merchant primarily selling shoes or other footwear",
    },
)

# A legacy combined category cannot safely be interpreted as footwear. Mapping it
# to the broad fashion bucket prevents historical or stale model output from
# contaminating shoe-specific analytics.
LEGACY_ITEM_CATEGORY_ALIASES: dict[str, str] = {
    "clothing_shoes": "fashion_unknown",
}
LEGACY_MERCHANT_CATEGORY_ALIASES: dict[str, str] = {
    "clothing_shoes": "fashion_retail",
}


def normalize_taxonomy_key(value: Any) -> str:
    """Normalize a model-supplied taxonomy key without adding semantic meaning."""

    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def canonical_item_category_key(value: Any) -> str:
    """Return the v2 item key, safely degrading the overloaded v1 fashion key."""

    key = normalize_taxonomy_key(value)
    return LEGACY_ITEM_CATEGORY_ALIASES.get(key, key)


def canonical_merchant_category_key(value: Any) -> str:
    """Return the v2 merchant key, including the safe legacy merchant alias."""

    key = normalize_taxonomy_key(value)
    return LEGACY_MERCHANT_CATEGORY_ALIASES.get(key, key)


def fashion_category_path(category_key: Any) -> str | None:
    """Resolve the canonical hierarchical path for a v2 fashion item key."""

    key = canonical_item_category_key(category_key)
    for row in FASHION_ITEM_TAXONOMY:
        if row["key"] == key:
            return row["path"]
    return None


__all__ = [
    "FASHION_ITEM_TAXONOMY",
    "FASHION_MERCHANT_TAXONOMY",
    "ITEM_TAXONOMY_VERSION",
    "LEGACY_ITEM_CATEGORY_ALIASES",
    "LEGACY_MERCHANT_CATEGORY_ALIASES",
    "MERCHANT_TAXONOMY_VERSION",
    "SPECIFIC_FASHION_ITEM_KEYS",
    "canonical_item_category_key",
    "canonical_merchant_category_key",
    "fashion_category_path",
    "normalize_taxonomy_key",
]
