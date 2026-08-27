#!/usr/bin/env python3
"""LLM-first item categorization for already extracted receipts.

This module intentionally runs after extraction/reconciliation. It never changes
receipt math, line totals, quantities, source_line_ids, payments, taxes, or the
validation decision. It only appends category metadata to item copies and writes
separate categorization artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    LlmGateway,
)
from receipt_intelligence.domain.categorization_taxonomy import (
    FASHION_ITEM_TAXONOMY,
    FASHION_MERCHANT_TAXONOMY,
    ITEM_TAXONOMY_VERSION,
    MERCHANT_TAXONOMY_VERSION,
    SPECIFIC_FASHION_ITEM_KEYS,
    canonical_item_category_key,
    canonical_merchant_category_key,
    normalize_taxonomy_key,
)
from receipt_intelligence.prompts import render_prompt_template

CATEGORY_SCHEMA_VERSION = "v14_14_item_categories_2"

CATEGORY_TAXONOMY: list[dict[str, str]] = [
    {
        "key": "groceries_food",
        "group": "Food & Groceries",
        "description": "General packaged food or ingredients",
    },
    {
        "key": "groceries_fruit_veg",
        "group": "Food & Groceries",
        "description": "Fruit, vegetables, fresh produce",
    },
    {
        "key": "groceries_bakery",
        "group": "Food & Groceries",
        "description": "Bread, bakery, pastry, grains",
    },
    {
        "key": "groceries_dairy_eggs",
        "group": "Food & Groceries",
        "description": "Milk, cheese, yoghurt, eggs, dairy alternatives",
    },
    {
        "key": "groceries_meat_fish",
        "group": "Food & Groceries",
        "description": "Meat, fish, cold cuts, seafood",
    },
    {
        "key": "beverages",
        "group": "Food & Groceries",
        "description": "Water, soft drinks, juice, coffee, tea; no alcohol-specific inference unless explicit",
    },
    {
        "key": "deposit_pfand",
        "group": "Deposits & Adjustments",
        "description": "Bottle/can deposit, Pfand charged on bottles/cans",
    },
    {
        "key": "deposit_refund",
        "group": "Deposits & Adjustments",
        "description": "Bottle/can deposit return, Leergut refund, negative Pfand adjustment",
    },
    {
        "key": "discount_coupon",
        "group": "Deposits & Adjustments",
        "description": "Discounts, coupons, promotions, rebates; not bottle deposit/refund",
    },
    {
        "key": "household_cleaning",
        "group": "Household",
        "description": "Cleaning supplies, detergent, dish tabs, paper towels",
    },
    {
        "key": "household_goods",
        "group": "Household",
        "description": "General home consumables and household objects",
    },
    {
        "key": "personal_care",
        "group": "Personal Care",
        "description": "Shampoo, toothpaste, soap, deodorant, hygiene",
    },
    {
        "key": "cosmetics",
        "group": "Personal Care",
        "description": "Makeup, beauty products, perfume, skincare cosmetics",
    },
    {
        "key": "baby_child",
        "group": "Family",
        "description": "Baby/child products, diapers, toys, child care",
    },
    {
        "key": "pharmacy_health",
        "group": "Health",
        "description": "Medication-like OTC, pharmacy, health products",
    },
    *FASHION_ITEM_TAXONOMY,
    {
        "key": "electronics",
        "group": "Electronics",
        "description": "Electronics, batteries, cables, devices",
    },
    {
        "key": "home_furniture",
        "group": "Home & Furniture",
        "description": "Furniture, home decoration, kitchenware, IKEA-like home articles",
    },
    {"key": "fuel", "group": "Mobility", "description": "Petrol, diesel, fuel, charging"},
    {
        "key": "restaurant_cafe",
        "group": "Restaurants & Cafes",
        "description": "Restaurant, takeaway, cafe items, prepared meals eaten out",
    },
    {
        "key": "transport_parking",
        "group": "Mobility",
        "description": "Tickets, parking, transport fees",
    },
    {"key": "services", "group": "Services", "description": "Service charges, repair, fees"},
    {"key": "other", "group": "Other", "description": "Known item but no better category"},
    {"key": "unknown", "group": "Unknown", "description": "Insufficient evidence"},
]

MERCHANT_TAXONOMY: list[dict[str, str]] = [
    {"key": "grocery_store", "description": "Supermarket, grocery or food retailer"},
    {"key": "restaurant_cafe", "description": "Restaurant, takeaway, cafe or prepared-food seller"},
    {"key": "bakery", "description": "Bakery or pastry shop"},
    {"key": "pharmacy_health", "description": "Pharmacy, drugstore or health retailer"},
    *FASHION_MERCHANT_TAXONOMY,
    {"key": "home_furniture", "description": "Furniture, home goods or household retailer"},
    {"key": "electronics", "description": "Electronics, appliance or device retailer"},
    {"key": "fuel", "description": "Fuel station or vehicle charging merchant"},
    {"key": "transport_parking", "description": "Transport, ticketing or parking provider"},
    {"key": "services", "description": "Service, repair or professional-service provider"},
    {"key": "general_retail", "description": "General or mixed non-specialist retailer"},
    {"key": "unknown", "description": "Insufficient merchant evidence"},
]

TAXONOMY_BY_KEY = {row["key"]: row for row in CATEGORY_TAXONOMY}
VALID_CATEGORY_KEYS = set(TAXONOMY_BY_KEY)
MERCHANT_TAXONOMY_BY_KEY = {row["key"]: row for row in MERCHANT_TAXONOMY}
VALID_MERCHANT_CATEGORY_KEYS = set(MERCHANT_TAXONOMY_BY_KEY)

CATEGORY_REVIEW_CONFIDENCE_THRESHOLD = 0.80
NOISY_TEXT_CONFIDENCE_CAP = 0.75
AMBIGUOUS_REASON_CONFIDENCE_CAP = 0.75
CONTEXTUAL_CONFIDENCE_CAP = 0.80
INCOMPLETE_TEXT_CONFIDENCE_CAP = 0.60
UNSUPPORTED_INFERENCE_CONFIDENCE_CAP = 0.60
UNKNOWN_CATEGORY_CONFIDENCE_CAP = 0.35

AMBIGUOUS_REASON_PATTERNS = [
    r"\blikely\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bassum",
    r"\bguess",
    r"\bunclear\b",
    r"\bambiguous\b",
    r"\bvague\b",
    r"\btruncated\b",
    r"\babbreviat",
    r"\bocr\b",
    r"not enough",
    r"insufficient",
    r"could be",
    r"may be",
    r"might be",
    r"wahrscheinlich",
    r"vermutlich",
    r"möglicherweise",
    r"moeglicherweise",
    r"unklar",
    r"nicht eindeutig",
    r"könnte",
    r"koennte",
    r"scheint",
    r"abgekürzt",
    r"abgekuerzt",
]

SEMANTIC_EXPANSION_PATTERNS = [
    r"\brefers to\b",
    r"\bmeans\b",
    r"\bshort for\b",
    r"\babbreviation for\b",
    r"\bis a type of\b",
    r"\bstands for\b",
    r"\bbedeutet\b",
    r"\bsteht für\b",
    r"\bsteht fuer\b",
    r"\bkurz für\b",
    r"\bkurz fuer\b",
]

VALID_TEXT_CERTAINTY = {"explicit", "contextual", "incomplete_or_unfamiliar", "ambiguous"}


NOISY_TEXT_MARKERS = [
    "�",
    "|",
    "§",
    "#",
    "@",
    "~",
    "^",
    "_",
    "<",
    ">",
    "{",
    "}",
    "[",
    "]",
]

COMMON_SHORT_PRODUCT_TOKENS = {
    "BIO",
    "XL",
    "XXL",
    "L",
    "M",
    "S",
    "XS",
    "2ER",
    "3ER",
    "4ER",
    "6ER",
    "TK",
    "WC",
    "SB",
    "FH",
}


def _tokenize_description(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]


def _is_code_like_description(text: str) -> bool:
    clean = re.sub(r"\s+", "", text or "")
    if not clean:
        return True
    digits = sum(ch.isdigit() for ch in clean)
    letters = sum(ch.isalpha() for ch in clean)
    punct = sum((not ch.isalnum()) for ch in clean)
    # Long numeric/article codes or mixed SKU strings should not receive high category confidence.
    if re.search(r"\d{5,}", clean):
        return True
    if digits >= 4 and digits / max(1, len(clean)) >= 0.30:
        return True
    if punct >= 3 and punct / max(1, len(clean)) >= 0.20:
        return True
    if letters == 0 and digits > 0:
        return True
    # Examples: 8043.00.549.B.01, ART123-45, 102.515.88
    if re.search(r"(?:\d+[\./-]){2,}\d+", clean):
        return True
    return False


def _looks_ocr_noisy_or_ambiguous_description(description: Any) -> tuple[bool, list[str]]:
    text = _safe_text(description, 240)
    reasons: list[str] = []
    if not text:
        return True, ["empty_description"]
    tokens = _tokenize_description(text)
    if len(text) <= 3:
        reasons.append("very_short_description")
    if len(tokens) == 1 and len(text) <= 5:
        reasons.append("single_short_token")
    if any(marker in text for marker in NOISY_TEXT_MARKERS):
        reasons.append("contains_ocr_marker")
    if _is_code_like_description(text):
        reasons.append("code_like_or_numeric_description")
    if text.endswith((".", ",", ":", ";", "-", "/")):
        reasons.append("truncated_or_punctuation_ending")
    if re.search(r"\b[A-Za-zÄÖÜäöüß]{1,2}\.$", text):
        reasons.append("abbreviated_last_token")
    if (
        len(tokens) >= 2
        and sum(
            1
            for t in tokens
            if len(re.sub(r"[^A-Za-zÄÖÜäöüß]", "", t)) <= 2
            and t.upper() not in COMMON_SHORT_PRODUCT_TOKENS
        )
        >= 2
    ):
        reasons.append("many_short_tokens")
    if tokens:
        last_alpha = re.sub(r"[^A-Za-zÄÖÜäöüß]", "", tokens[-1]).upper()
        if (
            2 <= len(last_alpha) <= 3
            and last_alpha not in COMMON_SHORT_PRODUCT_TOKENS
            and len(text) > 8
        ):
            reasons.append("truncated_last_token")
    return bool(reasons), reasons


def _looks_ambiguous_reason(reason: Any) -> tuple[bool, list[str]]:
    text = _safe_text(reason, 400).lower()
    reasons: list[str] = []
    if not text:
        reasons.append("missing_category_reason")
    for pattern in AMBIGUOUS_REASON_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            reasons.append(f"ambiguous_reason:{pattern}")
            break
    return bool(reasons), reasons


def _looks_like_semantic_expansion(reason: Any) -> bool:
    text = _safe_text(reason, 400).lower()
    return any(
        re.search(pattern, text, flags=re.IGNORECASE) for pattern in SEMANTIC_EXPANSION_PATTERNS
    )


def _normalize_text_certainty(value: Any) -> str:
    certainty = str(value or "contextual").strip().lower().replace("-", "_").replace(" ", "_")
    return certainty if certainty in VALID_TEXT_CERTAINTY else "contextual"


def _normalize_evidence_terms(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    terms: list[str] = []
    for item in value[:8]:
        term = _safe_text(item, 80)
        if term and term not in terms:
            terms.append(term)
    return terms


def _unsupported_evidence_terms(
    item: dict[str, Any], evidence_terms: list[str], context_text: str = ""
) -> list[str]:
    if not evidence_terms:
        return []
    source = (
        " ".join(
            _safe_text(item.get(field), 300)
            for field in ("product_description", "description", "raw_description", "line_note")
            if item.get(field)
        )
        + " "
        + _safe_text(context_text, 600)
    ).casefold()
    return [term for term in evidence_terms if term.casefold() not in source]


def calibrate_category_assignment(
    *,
    item: dict[str, Any] | None,
    category_key: str,
    confidence: float,
    reason: Any,
    source: str,
    text_certainty: Any = "contextual",
    evidence_terms: list[str] | None = None,
    context_text: str = "",
) -> dict[str, Any]:
    """Cap category confidence for OCR-noisy/ambiguous cases and mark review need.

    The LLM remains the first categorizer. This function does not change the chosen
    category. It only calibrates confidence and adds a review flag so downstream
    reporting can separate strong categories from weak guesses.
    """
    item = item or {}
    original_conf = _normalize_confidence(confidence, 0.0)
    calibrated = original_conf
    review_reasons: list[str] = []

    desc = item.get("description")
    noisy, noisy_reasons = _looks_ocr_noisy_or_ambiguous_description(desc)
    if noisy:
        calibrated = min(calibrated, NOISY_TEXT_CONFIDENCE_CAP)
        review_reasons.extend(noisy_reasons)

    ambiguous, ambiguous_reasons = _looks_ambiguous_reason(reason)
    if ambiguous:
        calibrated = min(calibrated, AMBIGUOUS_REASON_CONFIDENCE_CAP)
        review_reasons.extend(ambiguous_reasons)

    certainty = _normalize_text_certainty(text_certainty)
    if certainty != "explicit":
        calibrated = min(calibrated, CONTEXTUAL_CONFIDENCE_CAP)
        if original_conf > CONTEXTUAL_CONFIDENCE_CAP:
            review_reasons.append("non_explicit_text_cannot_receive_high_confidence")
    if certainty in {"incomplete_or_unfamiliar", "ambiguous"}:
        calibrated = min(calibrated, INCOMPLETE_TEXT_CONFIDENCE_CAP)
        review_reasons.append(f"text_certainty:{certainty}")

    normalized_evidence_terms = _normalize_evidence_terms(evidence_terms or [])
    unsupported_terms = _unsupported_evidence_terms(
        item, normalized_evidence_terms, context_text=context_text
    )
    if unsupported_terms:
        calibrated = min(calibrated, UNSUPPORTED_INFERENCE_CONFIDENCE_CAP)
        review_reasons.append("unsupported_category_evidence_terms")

    if _looks_like_semantic_expansion(reason):
        calibrated = min(calibrated, UNSUPPORTED_INFERENCE_CONFIDENCE_CAP)
        review_reasons.append("semantic_expansion_not_explicit_in_receipt_text")

    if category_key == "unknown":
        calibrated = min(calibrated, UNKNOWN_CATEGORY_CONFIDENCE_CAP)
        review_reasons.append("unknown_category")

    if source.startswith("fallback_"):
        review_reasons.append("fallback_category_source")

    calibrated = round(max(0.0, min(1.0, calibrated)), 3)
    if calibrated < CATEGORY_REVIEW_CONFIDENCE_THRESHOLD:
        review_reasons.append("low_calibrated_confidence")

    # Deduplicate while preserving order.
    deduped: list[str] = []
    for r in review_reasons:
        if r not in deduped:
            deduped.append(r)

    return {
        "category_confidence": calibrated,
        "category_confidence_raw": original_conf,
        "category_confidence_calibrated": calibrated,
        "category_review_required": bool(deduped),
        "category_review_reasons": deduped,
        "category_text_certainty": certainty,
        "category_evidence_terms": normalized_evidence_terms,
        "category_unsupported_evidence_terms": unsupported_terms,
    }


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _normalize_confidence(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    return round(max(0.0, min(1.0, v)), 3)


def _safe_text(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _taxonomy_text() -> str:
    lines = ["key|group|path|meaning"]
    for row in CATEGORY_TAXONOMY:
        path = row.get("path") or row["key"]
        lines.append(f"{row['key']}|{row['group']}|{path}|{row['description']}")
    return "\n".join(lines)


def _merchant_taxonomy_text() -> str:
    lines = ["key|meaning"]
    for row in MERCHANT_TAXONOMY:
        lines.append(f"{row['key']}|{row['description']}")
    return "\n".join(lines)


def _items_for_prompt(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(receipt.get("items") or []):
        if not isinstance(item, dict):
            continue
        product_text = (
            item.get("product_description")
            or item.get("clean_description")
            or item.get("normalized_name")
            or item.get("description")
        )
        noisy, noisy_reasons = _looks_ocr_noisy_or_ambiguous_description(product_text)
        out.append(
            {
                "item_index": idx,
                "description": _safe_text(product_text, 220),
                "product_description": _safe_text(product_text, 220),
                "raw_description": _safe_text(
                    item.get("raw_description") or item.get("description"), 260
                ),
                "line_note": _safe_text(item.get("line_note"), 220)
                if item.get("line_note")
                else None,
                "promotion_note": _safe_text(item.get("promotion_note"), 220)
                if item.get("promotion_note")
                else None,
                "quantity": item.get("quantity"),
                "unit": item.get("unit"),
                "unit_price": item.get("unit_price"),
                "original_price": item.get("original_price"),
                "discount_amount": item.get("discount_amount"),
                "line_total": item.get("line_total"),
                "tax_rate": item.get("tax_rate"),
                "parser_item_type": item.get("category"),
                "notes": _safe_text(item.get("notes"), 220) if item.get("notes") else None,
                "text_quality": {
                    "potentially_incomplete_or_noisy": noisy,
                    "signals": noisy_reasons,
                },
            }
        )
    return out


def _categorization_output_schema() -> dict[str, Any]:
    """Return the formal JSON Schema embedded in the categorization prompt."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": CATEGORY_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "taxonomy_version",
            "merchant_taxonomy_version",
            "merchant_classification",
            "items",
            "warnings",
        ],
        "properties": {
            "schema_version": {"const": CATEGORY_SCHEMA_VERSION},
            "taxonomy_version": {"const": ITEM_TAXONOMY_VERSION},
            "merchant_taxonomy_version": {"const": MERCHANT_TAXONOMY_VERSION},
            "merchant_classification": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category_key", "confidence", "reason"],
                "properties": {
                    "category_key": {
                        "type": "string",
                        "enum": sorted(VALID_MERCHANT_CATEGORY_KEYS),
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "item_index",
                        "category_key",
                        "confidence",
                        "text_certainty",
                        "evidence_terms",
                        "reason",
                    ],
                    "properties": {
                        "item_index": {"type": "integer", "minimum": 0},
                        "category_key": {
                            "type": "string",
                            "enum": sorted(VALID_CATEGORY_KEYS),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "text_certainty": {
                            "type": "string",
                            "enum": sorted(VALID_TEXT_CERTAINTY),
                        },
                        "evidence_terms": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string"},
                        },
                        "reason": {"type": "string"},
                    },
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }


def _categorization_envelope_warnings(obj: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if obj.get("schema_version") != CATEGORY_SCHEMA_VERSION:
        warnings.append(
            "Categorizer returned an unexpected schema_version; deterministic coercion applied."
        )
    if obj.get("taxonomy_version") != ITEM_TAXONOMY_VERSION:
        warnings.append(
            "Categorizer returned an unexpected taxonomy_version; v2 taxonomy enforcement applied."
        )
    if obj.get("merchant_taxonomy_version") != MERCHANT_TAXONOMY_VERSION:
        warnings.append(
            "Categorizer returned an unexpected merchant_taxonomy_version; "
            "v2 merchant taxonomy enforcement applied."
        )
    return warnings


def build_categorization_prompt(receipt: dict[str, Any]) -> str:
    merchant = receipt.get("merchant") if isinstance(receipt.get("merchant"), dict) else {}
    context = {
        "merchant_name": merchant.get("name"),
        "merchant_address": merchant.get("address"),
        "date": receipt.get("date"),
        "currency": receipt.get("currency") or "EUR",
        "totals": receipt.get("totals") or {},
        "parser_validation_decision": (
            (receipt.get("validation") or {}).get("import_decision")
            if isinstance(receipt.get("validation"), dict)
            else None
        ),
    }
    items = _items_for_prompt(receipt)
    schema = _categorization_output_schema()
    return render_prompt_template(
        "item_categorization.txt",
        TAXONOMY_TEXT=_taxonomy_text(),
        MERCHANT_TAXONOMY_TEXT=_merchant_taxonomy_text(),
        SCHEMA_JSON=json.dumps(schema, ensure_ascii=False, indent=2),
        CONTEXT_JSON=json.dumps(context, ensure_ascii=False, indent=2),
        ITEMS_JSON=json.dumps(items, ensure_ascii=False, indent=2),
    )


def _coerce_merchant_classification(
    obj: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    raw = obj.get("merchant_classification") if isinstance(obj, dict) else None
    if not isinstance(raw, dict):
        warnings.append("LLM categorizer returned no merchant_classification object.")
        raw = {}
    raw_key = normalize_taxonomy_key(raw.get("category_key") or "unknown")
    key = canonical_merchant_category_key(raw_key)
    if key != raw_key:
        warnings.append(
            f"Mapped legacy merchant category_key '{raw_key}' to taxonomy v2 key '{key}'."
        )
    if key not in VALID_MERCHANT_CATEGORY_KEYS:
        warnings.append(f"Unknown merchant category_key '{key}'; using unknown.")
        key = "unknown"
    return (
        {
            "category_key": key,
            "confidence": _normalize_confidence(raw.get("confidence"), 0.0),
            "reason": _safe_text(raw.get("reason"), 300),
            "source": "llm_first",
            "taxonomy_version": MERCHANT_TAXONOMY_VERSION,
        },
        warnings,
    )


def _coerce_categories(
    obj: dict[str, Any],
    original_items: list[dict[str, Any]],
    *,
    merchant_context_text: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    item_count = len(original_items)
    raw_items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(raw_items, list):
        warnings.append("LLM categorizer returned no items array.")
        raw_items = []
    by_index: dict[int, dict[str, Any]] = {}
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            idx = int(raw.get("item_index"))
        except Exception:
            warnings.append(f"Skipping category row without valid item_index: {raw}")
            continue
        if idx < 0 or idx >= item_count:
            warnings.append(f"Skipping category row with out-of-range item_index={idx}.")
            continue
        raw_key = normalize_taxonomy_key(raw.get("category_key") or "unknown")
        key = canonical_item_category_key(raw_key)
        if key != raw_key:
            warnings.append(
                f"Mapped legacy category_key '{raw_key}' to taxonomy v2 key '{key}' "
                f"for item_index={idx}."
            )
        if key not in VALID_CATEGORY_KEYS:
            warnings.append(f"Unknown category_key '{key}' for item_index={idx}; using unknown.")
            key = "unknown"
        tax = TAXONOMY_BY_KEY[key]
        reason = _safe_text(raw.get("reason"), 300)
        source = "llm_first"
        confidence = _normalize_confidence(raw.get("confidence"), 0.0)
        text_certainty = _normalize_text_certainty(raw.get("text_certainty"))
        evidence_terms = _normalize_evidence_terms(raw.get("evidence_terms"))
        if key in SPECIFIC_FASHION_ITEM_KEYS and text_certainty != "explicit":
            warnings.append(
                f"Downgraded contextual fashion subtype '{key}' to fashion_unknown "
                f"for item_index={idx}."
            )
            key = "fashion_unknown"
            tax = TAXONOMY_BY_KEY[key]
        calibration = calibrate_category_assignment(
            item=original_items[idx] if idx < len(original_items) else {},
            category_key=key,
            confidence=confidence,
            reason=reason,
            source=source,
            text_certainty=text_certainty,
            evidence_terms=evidence_terms,
            context_text=merchant_context_text,
        )
        by_index[idx] = {
            "item_index": idx,
            "category_key": key,
            "category_group": tax["group"],
            "category_path": str(tax.get("path") or key),
            "category_taxonomy_version": ITEM_TAXONOMY_VERSION,
            "category_confidence": calibration["category_confidence"],
            "category_confidence_raw": calibration["category_confidence_raw"],
            "category_confidence_calibrated": calibration["category_confidence_calibrated"],
            "category_review_required": calibration["category_review_required"],
            "category_review_reasons": calibration["category_review_reasons"],
            "category_text_certainty": calibration["category_text_certainty"],
            "category_evidence_terms": calibration["category_evidence_terms"],
            "category_unsupported_evidence_terms": calibration[
                "category_unsupported_evidence_terms"
            ],
            "category_reason": reason,
            "category_source": source,
        }
    categories: list[dict[str, Any]] = []
    for idx in range(item_count):
        if idx in by_index:
            categories.append(by_index[idx])
        else:
            warnings.append(f"Missing category for item_index={idx}; using unknown.")
            fallback_reason = "LLM did not return a category for this item."
            calibration = calibrate_category_assignment(
                item=original_items[idx] if idx < len(original_items) else {},
                category_key="unknown",
                confidence=0.0,
                reason=fallback_reason,
                source="fallback_unknown",
                context_text=merchant_context_text,
            )
            categories.append(
                {
                    "item_index": idx,
                    "category_key": "unknown",
                    "category_group": TAXONOMY_BY_KEY["unknown"]["group"],
                    "category_path": "unknown",
                    "category_taxonomy_version": ITEM_TAXONOMY_VERSION,
                    "category_confidence": calibration["category_confidence"],
                    "category_confidence_raw": calibration["category_confidence_raw"],
                    "category_confidence_calibrated": calibration["category_confidence_calibrated"],
                    "category_review_required": calibration["category_review_required"],
                    "category_review_reasons": calibration["category_review_reasons"],
                    "category_text_certainty": calibration["category_text_certainty"],
                    "category_evidence_terms": calibration["category_evidence_terms"],
                    "category_unsupported_evidence_terms": calibration[
                        "category_unsupported_evidence_terms"
                    ],
                    "category_reason": fallback_reason,
                    "category_source": "fallback_unknown",
                }
            )
    return categories, warnings


def merge_categories_into_receipt(
    receipt: dict[str, Any],
    categories: list[dict[str, Any]],
    *,
    status: str,
    merchant_classification: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    duration_seconds: float | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    out = deepcopy(receipt)
    items = out.get("items") if isinstance(out.get("items"), list) else []
    for cat in categories:
        idx = cat.get("item_index")
        if isinstance(idx, int) and 0 <= idx < len(items) and isinstance(items[idx], dict):
            items[idx]["category_key"] = cat.get("category_key")
            items[idx]["category_group"] = cat.get("category_group")
            items[idx]["category_path"] = cat.get("category_path")
            items[idx]["category_taxonomy_version"] = (
                cat.get("category_taxonomy_version") or ITEM_TAXONOMY_VERSION
            )
            items[idx]["category_confidence"] = cat.get("category_confidence")
            items[idx]["category_confidence_raw"] = cat.get("category_confidence_raw")
            items[idx]["category_confidence_calibrated"] = cat.get("category_confidence_calibrated")
            items[idx]["category_review_required"] = bool(cat.get("category_review_required"))
            items[idx]["category_review_reasons"] = cat.get("category_review_reasons") or []
            items[idx]["category_text_certainty"] = cat.get("category_text_certainty")
            items[idx]["category_evidence_terms"] = cat.get("category_evidence_terms") or []
            items[idx]["category_unsupported_evidence_terms"] = (
                cat.get("category_unsupported_evidence_terms") or []
            )
            items[idx]["category_source"] = cat.get("category_source")
            items[idx]["category_reason"] = cat.get("category_reason")
    out["items"] = items
    merchant_classification = merchant_classification or {
        "category_key": "unknown",
        "confidence": 0.0,
        "reason": "Merchant classification unavailable.",
        "source": "fallback_unknown",
    }
    merchant = out.get("merchant") if isinstance(out.get("merchant"), dict) else {}
    merchant = dict(merchant)
    merchant["category_key"] = merchant_classification.get("category_key")
    merchant["category_confidence"] = merchant_classification.get("confidence")
    merchant["category_reason"] = merchant_classification.get("reason")
    merchant["category_source"] = merchant_classification.get("source")
    merchant["category_taxonomy_version"] = (
        merchant_classification.get("taxonomy_version") or MERCHANT_TAXONOMY_VERSION
    )
    out["merchant"] = merchant
    out["categorization"] = {
        "schema_version": CATEGORY_SCHEMA_VERSION,
        "taxonomy_version": ITEM_TAXONOMY_VERSION,
        "merchant_taxonomy_version": MERCHANT_TAXONOMY_VERSION,
        "app_version": get_app_version(),
        "status": status,
        "mode": "llm_first",
        "model": model,
        "merchant_classification": merchant_classification,
        "item_count": len(items),
        "categorized_count": sum(
            1 for item in items if isinstance(item, dict) and item.get("category_key")
        ),
        "category_review_count": sum(
            1 for item in items if isinstance(item, dict) and item.get("category_review_required")
        ),
        "review_confidence_threshold": CATEGORY_REVIEW_CONFIDENCE_THRESHOLD,
        "confidence_calibration": {
            "enabled": True,
            "noisy_text_cap": NOISY_TEXT_CONFIDENCE_CAP,
            "ambiguous_reason_cap": AMBIGUOUS_REASON_CONFIDENCE_CAP,
            "contextual_confidence_cap": CONTEXTUAL_CONFIDENCE_CAP,
            "incomplete_text_cap": INCOMPLETE_TEXT_CONFIDENCE_CAP,
            "unsupported_inference_cap": UNSUPPORTED_INFERENCE_CONFIDENCE_CAP,
            "unknown_category_cap": UNKNOWN_CATEGORY_CONFIDENCE_CAP,
        },
        "warnings": warnings or [],
        "duration_seconds": duration_seconds,
        "taxonomy": [
            {
                "key": r["key"],
                "group": r["group"],
                "path": r.get("path") or r["key"],
            }
            for r in CATEGORY_TAXONOMY
        ],
    }
    return out


def unknown_categories_for_receipt(
    receipt: dict[str, Any], reason: str = "categorization disabled or failed"
) -> list[dict[str, Any]]:
    items = receipt.get("items") if isinstance(receipt.get("items"), list) else []
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        calibration = calibrate_category_assignment(
            item=item if isinstance(item, dict) else {},
            category_key="unknown",
            confidence=0.0,
            reason=reason,
            source="fallback_unknown",
        )
        out.append(
            {
                "item_index": idx,
                "category_key": "unknown",
                "category_group": TAXONOMY_BY_KEY["unknown"]["group"],
                "category_path": "unknown",
                "category_taxonomy_version": ITEM_TAXONOMY_VERSION,
                "category_confidence": calibration["category_confidence"],
                "category_confidence_raw": calibration["category_confidence_raw"],
                "category_confidence_calibrated": calibration["category_confidence_calibrated"],
                "category_review_required": calibration["category_review_required"],
                "category_review_reasons": calibration["category_review_reasons"],
                "category_text_certainty": calibration["category_text_certainty"],
                "category_evidence_terms": calibration["category_evidence_terms"],
                "category_unsupported_evidence_terms": calibration[
                    "category_unsupported_evidence_terms"
                ],
                "category_reason": reason,
                "category_source": "fallback_unknown",
            }
        )
    return out


def recalibrate_existing_categorized_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Re-apply V14.14.1 category confidence calibration to an existing categorized receipt.

    This does not call the LLM and does not change category_key/category_group. It is
    useful for backfilling category_review_required on V14.14 categorized JSON files.
    """
    out = deepcopy(receipt)
    items = out.get("items") if isinstance(out.get("items"), list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = canonical_item_category_key(item.get("category_key") or "unknown")
        if key not in VALID_CATEGORY_KEYS:
            key = "unknown"
        raw_conf = item.get("category_confidence_raw", item.get("category_confidence", 0.0))
        source = str(item.get("category_source") or "llm_first")
        calibration = calibrate_category_assignment(
            item=item,
            category_key=key,
            confidence=_normalize_confidence(raw_conf, 0.0),
            reason=item.get("category_reason"),
            source=source,
            text_certainty=item.get("category_text_certainty"),
            evidence_terms=item.get("category_evidence_terms") or [],
        )
        item["category_key"] = key
        item["category_group"] = TAXONOMY_BY_KEY[key]["group"]
        item["category_path"] = str(TAXONOMY_BY_KEY[key].get("path") or key)
        item["category_taxonomy_version"] = ITEM_TAXONOMY_VERSION
        item["category_confidence"] = calibration["category_confidence"]
        item["category_confidence_raw"] = calibration["category_confidence_raw"]
        item["category_confidence_calibrated"] = calibration["category_confidence_calibrated"]
        item["category_review_required"] = calibration["category_review_required"]
        item["category_review_reasons"] = calibration["category_review_reasons"]
        item["category_text_certainty"] = calibration["category_text_certainty"]
        item["category_evidence_terms"] = calibration["category_evidence_terms"]
        item["category_unsupported_evidence_terms"] = calibration[
            "category_unsupported_evidence_terms"
        ]
    out["items"] = items
    merchant = out.get("merchant") if isinstance(out.get("merchant"), dict) else {}
    if merchant:
        merchant = dict(merchant)
        merchant_key = canonical_merchant_category_key(merchant.get("category_key") or "unknown")
        if merchant_key not in VALID_MERCHANT_CATEGORY_KEYS:
            merchant_key = "unknown"
        merchant["category_key"] = merchant_key
        merchant["category_taxonomy_version"] = MERCHANT_TAXONOMY_VERSION
        out["merchant"] = merchant
    cat = out.get("categorization") if isinstance(out.get("categorization"), dict) else {}
    cat = dict(cat)
    cat["app_version"] = get_app_version()
    cat["schema_version"] = CATEGORY_SCHEMA_VERSION
    cat["taxonomy_version"] = ITEM_TAXONOMY_VERSION
    cat["merchant_taxonomy_version"] = MERCHANT_TAXONOMY_VERSION
    cat["category_review_count"] = sum(
        1 for item in items if isinstance(item, dict) and item.get("category_review_required")
    )
    cat["review_confidence_threshold"] = CATEGORY_REVIEW_CONFIDENCE_THRESHOLD
    cat["confidence_calibration"] = {
        "enabled": True,
        "noisy_text_cap": NOISY_TEXT_CONFIDENCE_CAP,
        "ambiguous_reason_cap": AMBIGUOUS_REASON_CONFIDENCE_CAP,
        "contextual_confidence_cap": CONTEXTUAL_CONFIDENCE_CAP,
        "incomplete_text_cap": INCOMPLETE_TEXT_CONFIDENCE_CAP,
        "unsupported_inference_cap": UNSUPPORTED_INFERENCE_CONFIDENCE_CAP,
        "unknown_category_cap": UNKNOWN_CATEGORY_CONFIDENCE_CAP,
    }
    out["categorization"] = cat
    return out


def categorize_receipt_items_llm(
    receipt: dict[str, Any],
    *,
    ollama_url: str,
    model: str,
    num_ctx: int = 16384,
    num_predict: int = 4096,
    keep_alive: str | None = None,
    timeout: float = 180.0,
    format_json: bool = True,
    llm_gateway: LlmGateway | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    item_count = len(receipt.get("items") or []) if isinstance(receipt.get("items"), list) else 0
    prompt = build_categorization_prompt(receipt)
    response_schema = _categorization_output_schema()
    raw = ""
    warnings: list[str] = []
    if item_count <= 0:
        categorized = merge_categories_into_receipt(
            receipt,
            [],
            status="skipped_no_items",
            merchant_classification={
                "category_key": "unknown",
                "confidence": 0.0,
                "reason": "No items available for merchant-context categorization.",
                "source": "fallback_unknown",
            },
            warnings=["No items to categorize."],
            duration_seconds=0.0,
            model=model,
        )
        return {
            "status": "skipped_no_items",
            "receipt": categorized,
            "prompt": prompt,
            "raw_output": raw,
            "categories": [],
            "merchant_classification": {
                "category_key": "unknown",
                "confidence": 0.0,
                "reason": "No items available for merchant-context categorization.",
                "source": "fallback_unknown",
            },
            "warnings": ["No items to categorize."],
            "duration_seconds": 0.0,
            "error": None,
        }
    try:
        if llm_gateway is None:
            from receipt_intelligence.adapters.llm import OllamaGateway

            llm_gateway = OllamaGateway(ollama_url)
        generation = llm_gateway.generate(
            GenerationRequest(
                model=model,
                prompt=prompt,
                operation="receipt_item_categorization",
                num_ctx=num_ctx,
                num_predict=num_predict,
                temperature=0.0,
                keep_alive=keep_alive,
                timeout_seconds=timeout,
                format_json=format_json,
                response_json_schema=response_schema if format_json else None,
            )
        )
        raw = generation.text
        parsed = parse_json_from_llm(
            generation,
            response_json_schema=response_schema,
        )
        warnings.extend(_categorization_envelope_warnings(parsed))
        original_items = [item for item in (receipt.get("items") or []) if isinstance(item, dict)]
        merchant_classification, merchant_warnings = _coerce_merchant_classification(parsed)
        merchant = receipt.get("merchant") if isinstance(receipt.get("merchant"), dict) else {}
        merchant_context_text = " ".join(
            str(value or "")
            for value in (
                merchant.get("name"),
                merchant.get("address"),
                merchant_classification.get("category_key"),
            )
        )
        categories, coercion_warnings = _coerce_categories(
            parsed,
            original_items,
            merchant_context_text=merchant_context_text,
        )
        warnings.extend(merchant_warnings)
        warnings.extend(coercion_warnings)
        status = "ok" if not warnings else "ok_with_warnings"
        duration = round(time.perf_counter() - started, 2)
        categorized = merge_categories_into_receipt(
            receipt,
            categories,
            status=status,
            merchant_classification=merchant_classification,
            warnings=warnings,
            duration_seconds=duration,
            model=model,
        )
        return {
            "status": status,
            "receipt": categorized,
            "prompt": prompt,
            "raw_output": raw,
            "categories": categories,
            "merchant_classification": merchant_classification,
            "warnings": warnings,
            "duration_seconds": duration,
            "error": None,
        }
    except Exception as exc:
        duration = round(time.perf_counter() - started, 2)
        err = f"{type(exc).__name__}: {exc}"
        warnings.append(err)
        categories = unknown_categories_for_receipt(receipt, "LLM categorization failed.")
        merchant_classification = {
            "category_key": "unknown",
            "confidence": 0.0,
            "reason": "LLM categorization failed.",
            "source": "fallback_unknown",
        }
        categorized = merge_categories_into_receipt(
            receipt,
            categories,
            status="error",
            merchant_classification=merchant_classification,
            warnings=warnings,
            duration_seconds=duration,
            model=model,
        )
        return {
            "status": "error",
            "receipt": categorized,
            "prompt": prompt,
            "raw_output": raw,
            "categories": categories,
            "merchant_classification": merchant_classification,
            "warnings": warnings,
            "duration_seconds": duration,
            "error": err,
        }


def write_categorization_artifacts(
    result: dict[str, Any], *, result_dir: Path, run_id: str
) -> dict[str, Path]:
    paths = {
        "categorization_prompt": result_dir / f"{run_id}_v14_14_categorization_prompt.txt",
        "categorization_raw": result_dir / f"{run_id}_v14_14_categorization_raw.txt",
        "categorization_result": result_dir / f"{run_id}_v14_14_categorization_result.json",
        "receipt_final_categorized": result_dir / f"{run_id}_receipt_final_categorized.json",
    }
    _write_text(paths["categorization_prompt"], result.get("prompt") or "")
    _write_text(paths["categorization_raw"], result.get("raw_output") or "")
    _save_json(
        paths["categorization_result"],
        {
            "schema_version": CATEGORY_SCHEMA_VERSION,
            "taxonomy_version": ITEM_TAXONOMY_VERSION,
            "merchant_taxonomy_version": MERCHANT_TAXONOMY_VERSION,
            "app_version": get_app_version(),
            "status": result.get("status"),
            "categories": result.get("categories") or [],
            "merchant_classification": result.get("merchant_classification") or {},
            "warnings": result.get("warnings") or [],
            "duration_seconds": result.get("duration_seconds"),
            "error": result.get("error"),
        },
    )
    _save_json(paths["receipt_final_categorized"], result.get("receipt") or {})
    aliases = {
        "latest_v14_14_categorization_prompt": (
            paths["categorization_prompt"],
            result_dir / "latest_v14_14_categorization_prompt.txt",
        ),
        "latest_v14_14_categorization_raw": (
            paths["categorization_raw"],
            result_dir / "latest_v14_14_categorization_raw.txt",
        ),
        "latest_v14_14_categorization_result": (
            paths["categorization_result"],
            result_dir / "latest_v14_14_categorization_result.json",
        ),
        "latest_receipt_final_categorized": (
            paths["receipt_final_categorized"],
            result_dir / "latest_receipt_final_categorized.json",
        ),
    }
    for key, (src, dst) in aliases.items():
        if src.exists():
            dst.write_bytes(src.read_bytes())
            paths[key] = dst
    return paths


def categorize_receipt_file(
    *,
    receipt_path: Path,
    out_dir: Path,
    run_id: str | None,
    ollama_url: str,
    model: str,
    num_ctx: int = 16384,
    num_predict: int = 4096,
    keep_alive: str | None = None,
    timeout: float = 180.0,
    format_json: bool = True,
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    rid = run_id or receipt_path.stem.replace("_receipt_final_reconciled", "")
    result = categorize_receipt_items_llm(
        receipt,
        ollama_url=ollama_url,
        model=model,
        num_ctx=num_ctx,
        num_predict=num_predict,
        keep_alive=keep_alive,
        timeout=timeout,
        format_json=format_json,
    )
    paths = write_categorization_artifacts(result, result_dir=out_dir, run_id=rid)
    return {
        "status": result.get("status"),
        "paths": {k: str(v) for k, v in paths.items()},
        "duration_seconds": result.get("duration_seconds"),
        "error": result.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Categorize an already extracted V14 receipt JSON with LLM-first item categorization."
    )
    parser.add_argument("receipt_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--keep-alive", default="")
    parser.add_argument("--no-format-json", action="store_true")
    args = parser.parse_args()
    summary = categorize_receipt_file(
        receipt_path=args.receipt_json,
        out_dir=args.out_dir,
        run_id=args.run_id,
        ollama_url=args.ollama_url,
        model=args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive or None,
        timeout=args.timeout,
        format_json=not args.no_format_json,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
