"""Normalization helpers shared by storage repositories and query planning."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from receipt_intelligence.domain.categorization_taxonomy import (
    canonical_item_category_key,
    fashion_category_path,
)
from receipt_intelligence.utils.text import normalize_text

_WORD_RE = re.compile(r"[\wäöüÄÖÜß]+", re.UNICODE)

CATEGORY_ALIASES: dict[str, list[str]] = {
    "personal_care/shampoo": [
        "shampoo",
        "hair shampoo",
        "haar shampoo",
        "haarpflege shampoo",
        "head shoulders",
        "head&shoulders",
        "h&s",
        "elvital",
        "fructis",
        "pantene",
        "dove shampoo",
        "nivea shampoo",
        "guhl",
        "schauma",
        "gliss kur",
    ],
    "personal_care/hygiene": [
        "hygiene",
        "personal care",
        "körperpflege",
        "hygieneartikel",
        "shampoo",
        "duschgel",
        "shower gel",
        "deo",
        "deodorant",
        "zahnpasta",
        "toothpaste",
        "seife",
        "soap",
        "rasierer",
        "toilet paper",
        "toilettenpapier",
        "tampon",
        "binden",
        "always",
        "elmex",
        "oral b",
        "nivea",
        "dove",
        "balea",
        "isana",
    ],
    "household/cleaning": [
        "cleaning supplies",
        "cleaner",
        "reiniger",
        "spülmittel",
        "waschmittel",
        "detergent",
        "domestos",
        "wc reiniger",
        "allzweckreiniger",
    ],
    "baby/baby_products": [
        "baby",
        "windeln",
        "diapers",
        "feuchttücher",
        "wipes",
        "hipp",
        "bebivita",
        "milupa",
        "aptamil",
    ],
    "pharmacy/medicine": [
        "medicine",
        "medication",
        "arznei",
        "apotheke",
        "tabletten",
        "ibuprofen",
        "paracetamol",
        "nasenspray",
    ],
}

MERCHANT_ALIASES: dict[str, list[str]] = {
    "dm": ["dm", "dm-drogerie", "dm drogerie", "dm-drogerie markt", "drogerie markt"],
    "rossmann": ["rossmann", "rossmann drogerie"],
    "rewe": ["rewe"],
    "aldi": ["aldi", "aldi süd", "aldi sued", "aldi nord"],
    "lidl": ["lidl"],
    "edeka": ["edeka"],
    "kaufland": ["kaufland"],
}

PARSER_ITEM_TYPES = {
    "item",
    "discount",
    "deposit",
    "refund",
    "fee",
    "tax",
    "subtotal",
    "total",
    "payment",
    "info",
    "unknown",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def tokenize(value: Any) -> list[str]:
    normalized = normalize_text(value)
    return [match.group(0) for match in _WORD_RE.finditer(normalized)]


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("€", "").replace("EUR", "").replace(" ", "")
    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    elif text.count(",") == 1 and text.count(".") >= 1:
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except Exception:
        return None


def as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def extract_item_description(item: dict[str, Any]) -> str:
    """Return product-identifying text, excluding adjacent promotion noise."""
    return str(
        first_present(
            item.get("product_description"),
            item.get("clean_description"),
            item.get("normalized_name"),
            item.get("description"),
            item.get("name"),
            item.get("raw_name"),
            item.get("text"),
            "",
        )
    )


def _is_parser_item_type(value: Any) -> bool:
    return normalize_text(value).replace(" ", "_") in PARSER_ITEM_TYPES


def parser_item_type_from_item(item: dict[str, Any]) -> str | None:
    explicit = first_present(
        item.get("parser_item_type"),
        item.get("receipt_row_type"),
        item.get("line_type"),
        item.get("category"),
    )
    return as_str(explicit)


def _specific_category_from_aliases(description: str) -> str | None:
    normalized = normalize_text(description)
    for category, aliases in CATEGORY_ALIASES.items():
        if any(normalize_text(alias) in normalized for alias in aliases):
            return category
    return None


def _canonical_spending_category(value: Any) -> str | None:
    text = as_str(value)
    if not text:
        return None
    key = canonical_item_category_key(text)
    return fashion_category_path(key) or text


def category_from_item(item: dict[str, Any], description: str) -> str | None:
    alias_category = _specific_category_from_aliases(description)
    if alias_category:
        return alias_category

    explicit = first_present(
        item.get("product_category"),
        item.get("spending_category"),
        item.get("analytics_category"),
        item.get("category_path"),
    )
    if explicit and not _is_parser_item_type(explicit):
        return _canonical_spending_category(explicit)

    group = as_str(item.get("category_group"))
    key = as_str(item.get("category_key"))
    if key and not _is_parser_item_type(key):
        return _canonical_spending_category(key)
    if group and not _is_parser_item_type(group):
        return group

    legacy = item.get("category")
    if legacy and not _is_parser_item_type(legacy):
        return _canonical_spending_category(legacy)
    return None


def normalize_merchant_name(value: Any) -> str | None:
    normalized = normalize_text(value)
    if not normalized:
        return None
    for canonical, aliases in MERCHANT_ALIASES.items():
        if any(normalize_text(alias) in normalized for alias in aliases):
            return canonical
    return normalized


def build_item_embedding_text(
    *,
    merchant_name: str | None,
    merchant_normalized: str | None,
    receipt_date: str | None,
    item: dict[str, Any],
    description: str,
    normalized_name: str | None,
    category: str | None,
    line_total: float | None,
    currency: str | None,
) -> str:
    aliases = CATEGORY_ALIASES.get(category or "", [])
    parts = [
        f"Merchant: {merchant_name or merchant_normalized or 'unknown'}",
        f"Date: {receipt_date or 'unknown'}",
        f"Product item: {description}",
        f"Raw item: {first_present(item.get('raw_description'), item.get('description'), description)}",
        f"Normalized item: {normalized_name or description}",
        f"Line note: {first_present(item.get('line_note'), item.get('promotion_note'), 'none')}",
        f"Category: {category or 'unknown'}",
        "Semantic description: "
        f"{first_present(item.get('semantic_description'), item.get('category_reason'), 'none')}",
        f"Parser row type: {parser_item_type_from_item(item) or 'unknown'}",
        f"Original price: {first_present(item.get('original_price'), item.get('gross_unit_price'), 'unknown')}",
        f"Discount amount: {first_present(item.get('discount_amount'), 'unknown')}",
        f"Tax code: {first_present(item.get('tax_code'), 'unknown')}",
        f"Aliases: {', '.join(aliases[:20])}",
        f"Price: {line_total if line_total is not None else 'unknown'} {currency or ''}".strip(),
    ]
    return "\n".join(parts)
