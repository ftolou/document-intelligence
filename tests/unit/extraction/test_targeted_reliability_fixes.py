"""Regression tests for the scoped v1.24.1 reliability fixes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from receipt_intelligence.extraction.categorization.items import (
    build_categorization_prompt,
    categorize_receipt_items_llm,
)
from receipt_intelligence.extraction.parsing.llm_parser import run_llm_main_parser
from receipt_intelligence.extraction.repair.item_order import sort_items_by_printed_order


def _valid_receipt_without_overall_confidence() -> dict:
    return {
        "schema_version": "v14_6_llm_receipt_1",
        "parse_status": "ok",
        "currency": "EUR",
        "merchant": {
            "name": "REWE",
            "address": None,
            "tax_id": None,
            "source_line_ids": ["line_001"],
        },
        "date": "2026-07-14",
        "time": "12:00:00",
        "items": [
            {
                "description": "MILCH",
                "line_total": 1.29,
                "category": "item",
                "source_line_ids": ["line_010"],
            }
        ],
        "taxes": [],
        "totals": {
            "subtotal": 1.29,
            "tax_total": None,
            "grand_total": 1.29,
            "paid_total": 1.29,
            "change": None,
            "source_line_ids": ["line_020"],
        },
        "payments": [],
        "unresolved_rows": [],
        "warnings": [],
    }


def test_missing_overall_confidence_does_not_trigger_full_llm_retry(tmp_path: Path) -> None:
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps({"image_width": 100, "image_height": 100, "words": [], "lines": []}),
        encoding="utf-8",
    )
    raw = json.dumps(_valid_receipt_without_overall_confidence())

    with patch(
        "receipt_intelligence.extraction.parsing.llm_parser.ollama_generate",
        return_value=raw,
    ) as generate:
        result = run_llm_main_parser(
            ocr_json_path=ocr_path,
            ollama_url="http://unused",
            model="test-model",
            json_retry_count=2,
        )

    assert generate.call_count == 1
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["status"] == "ok"
    assert result["receipt"]["overall_confidence"] == 0.6
    assert any("without an LLM retry" in warning for warning in result["receipt"]["warnings"])


def test_sequence_merge_preserves_printed_order_across_line_namespaces() -> None:
    region_sequence = [
        {"description": "APPLE", "source_line_ids": ["region_line_002"]},
        {"description": "CHEESE", "source_line_ids": ["region_line_004"]},
    ]
    table_sequence = [
        {"description": "APPLE", "source_line_ids": ["line_010"]},
        {"description": "BREAD", "source_line_ids": ["line_011"]},
        {"description": "CHEESE", "source_line_ids": ["line_012"]},
    ]
    mixed_items = [
        {"description": "CHEESE", "line_total": 3.0},
        {"description": "APPLE", "line_total": 1.0},
        {"description": "BREAD", "line_total": 2.0},
    ]

    ordered = sort_items_by_printed_order(
        mixed_items,
        sequences=[region_sequence, table_sequence],
    )

    assert [item["description"] for item in ordered] == ["APPLE", "BREAD", "CHEESE"]


def test_categorizer_caps_incomplete_semantic_expansion_without_dictionary() -> None:
    receipt = {
        "merchant": {"name": "REWE"},
        "currency": "EUR",
        "items": [
            {
                "description": "BIO TK SCHNITTLA",
                "product_description": "BIO TK SCHNITTLA",
                "raw_description": "BIO TK SCHNITTLA",
                "line_total": 0.79,
                "category": "item",
            }
        ],
    }
    model_output = {
        "schema_version": "v14_14_item_categories_1",
        "items": [
            {
                "item_index": 0,
                "description": "BIO TK SCHNITTLA",
                "category_key": "groceries_meat_fish",
                "category_group": "Food & Groceries",
                "confidence": 1.0,
                "text_certainty": "incomplete_or_unfamiliar",
                "evidence_terms": ["BIO", "TK", "SCHNITTLA"],
                "reason": "Schnittl refers to a meat product.",
            }
        ],
        "warnings": [],
    }

    with patch(
        "receipt_intelligence.extraction.categorization.items.ollama_generate",
        return_value=json.dumps(model_output),
    ):
        result = categorize_receipt_items_llm(
            receipt,
            ollama_url="http://unused",
            model="test-model",
        )

    item = result["receipt"]["items"][0]
    assert item["category_key"] == "groceries_meat_fish"
    assert item["category_confidence"] <= 0.60
    assert item["category_review_required"] is True
    assert "text_certainty:incomplete_or_unfamiliar" in item["category_review_reasons"]
    assert "semantic_expansion_not_explicit_in_receipt_text" in item["category_review_reasons"]
    assert "Never complete, translate, or invent" in build_categorization_prompt(receipt)


def test_categorizer_classifies_merchant_before_grouped_items() -> None:
    receipt = {
        "merchant": {"name": "Pizza Express MOD"},
        "currency": "EUR",
        "items": [
            {"description": "Bella", "line_total": 7.50, "category": "item"},
            {"description": "Rustika", "line_total": 7.50, "category": "item"},
        ],
    }
    model_output = {
        "schema_version": "v14_14_item_categories_1",
        "merchant_classification": {
            "category_key": "restaurant_cafe",
            "confidence": 0.96,
            "reason": "Pizza Express and the peer items indicate prepared food.",
        },
        "items": [
            {
                "item_index": 0,
                "description": "Bella",
                "category_key": "restaurant_cafe",
                "category_group": "Restaurants & Cafes",
                "confidence": 0.88,
                "text_certainty": "contextual",
                "evidence_terms": ["Pizza", "Bella"],
                "reason": "Restaurant merchant context and peer-item pattern.",
            },
            {
                "item_index": 1,
                "description": "Rustika",
                "category_key": "restaurant_cafe",
                "category_group": "Restaurants & Cafes",
                "confidence": 0.89,
                "text_certainty": "contextual",
                "evidence_terms": ["Pizza", "Rustika"],
                "reason": "Restaurant merchant context and peer-item pattern.",
            },
        ],
        "warnings": [],
    }

    with patch(
        "receipt_intelligence.extraction.categorization.items.ollama_generate",
        return_value=json.dumps(model_output),
    ):
        result = categorize_receipt_items_llm(
            receipt,
            ollama_url="http://unused",
            model="test-model",
        )

    categorized = result["receipt"]
    assert categorized["merchant"]["category_key"] == "restaurant_cafe"
    assert categorized["merchant"]["category_confidence"] == 0.96
    assert [item["category_key"] for item in categorized["items"]] == [
        "restaurant_cafe",
        "restaurant_cafe",
    ]
    prompt = build_categorization_prompt(receipt)
    assert "Merchant taxonomy" in prompt
    assert "classify the merchant" in prompt
