from __future__ import annotations

from receipt_intelligence.extraction.evidence.region_price_fusion import (
    build_region_item_price_candidates,
    region_price_candidates_to_prompt_text,
)
from receipt_intelligence.extraction.repair.line_price_fusion import (
    repair_receipt_line_prices,
)


def _line(
    line_id: str,
    text: str,
    *,
    x: float,
    y: float,
    confidence: float = 0.98,
    amount: float | None = None,
    role: str,
    damaged: bool = False,
) -> dict:
    return {
        "id": line_id,
        "text": text,
        "confidence": confidence,
        "xmin": x,
        "ymin": y,
        "xmax": x + 150,
        "ymax": y + 50,
        "x_center": x + 75,
        "y_center": y + 25,
        "amounts": [] if amount is None else [{"raw": text.split()[0], "value": amount}],
        "damaged_amount_candidate": (
            {"raw": text, "value": None, "status": "damaged_amount_token"}
            if damaged
            else None
        ),
        "role_hint": role,
    }


def _visual_evidence() -> dict:
    lines = [
        _line(
            "region_line_001",
            "KART. VORW. FESTK.",
            x=60,
            y=500,
            role="product_or_item_text",
        ),
        _line(
            "region_line_002",
            "1,99 B",
            x=800,
            y=505,
            amount=1.99,
            role="amount_only",
        ),
        _line(
            "region_line_003",
            "H-MILCH 3,8 %",
            x=60,
            y=600,
            role="product_or_item_text",
        ),
        _line(
            "region_line_004",
            "2,58 B",
            x=800,
            y=605,
            amount=2.58,
            role="amount_only",
        ),
        _line(
            "region_line_005",
            "2 Stk x",
            x=120,
            y=660,
            role="quantity_or_unit_price_note",
        ),
        _line(
            "region_line_006",
            "1,29",
            x=400,
            y=660,
            amount=1.29,
            role="amount_only",
        ),
        _line(
            "region_line_007",
            "3,35 A",
            x=800,
            y=780,
            amount=3.35,
            role="amount_only",
        ),
        _line(
            "region_line_008",
            "LUXUS TOILET. PAP",
            x=60,
            y=800,
            role="product_or_item_text",
        ),
        _line(
            "region_line_009",
            "0,6A",
            x=800,
            y=830,
            confidence=0.75,
            role="damaged_amount_candidate",
            damaged=True,
        ),
        _line(
            "region_line_010",
            "WATTEPADS MAXI",
            x=60,
            y=860,
            role="product_or_item_text",
        ),
        _line(
            "region_line_011",
            "3,99 A",
            x=800,
            y=900,
            amount=3.99,
            role="amount_only",
        ),
        _line(
            "region_line_012",
            "TORTENGLOCKE PIN",
            x=60,
            y=920,
            role="product_or_item_text",
        ),
        _line(
            "region_line_013",
            "AS-Zeit",
            x=60,
            y=1040,
            role="product_or_item_text",
        ),
        _line(
            "region_line_014",
            "32,22",
            x=800,
            y=1040,
            amount=32.22,
            role="amount_only",
        ),
    ]
    preferred = {
        "region_id": "region_00",
        "rows": [
            {
                "row_id": "region_line_001",
                "description_candidate": "KART. VORW. FESTK.",
                "amount": 1.99,
                "amount_raw": "1,99",
                "source_line_ids": ["region_line_001", "region_line_002"],
                "layout_confidence": 0.98,
                "evidence_source": "crop_ocr_right_amount_same_y_band",
            },
            {
                "row_id": "region_line_003",
                "description_candidate": "H-MILCH 3,8 %",
                "amount": 2.58,
                "amount_raw": "2,58",
                "source_line_ids": ["region_line_003", "region_line_004"],
                "layout_confidence": 0.97,
                "evidence_source": "crop_ocr_right_amount_same_y_band",
            },
        ],
        "quantity_note_links": [
            {
                "quantity_row_id": "region_line_005",
                "quantity_text": "2 Stk x",
                "linked_item_row_id": "region_line_003",
            }
        ],
        "unmatched_product_rows": [
            {"row_id": "region_line_008", "text": "LUXUS TOILET. PAP"},
            {"row_id": "region_line_010", "text": "WATTEPADS MAXI"},
        ],
    }
    return {
        "status": "ok",
        "region_reocr": {
            "status": "ok",
            "regions": [{"region_id": "region_00", "lines": lines}],
            "preferred_item_blocks": [preferred],
        },
    }


def _spatial_rows() -> list[dict]:
    return [
        {
            "row_id": "spatial_row_001",
            "line_id": "line_001",
            "text": "KART. VORW. FESTK.",
            "bbox": {"x": 0.06, "y": 0.25, "w": 0.4, "h": 0.025},
            "cells": [],
            "geometric_group_id": "geometry_group_001",
        },
        {
            "row_id": "spatial_row_002",
            "line_id": "line_002",
            "text": "H-MILCH 3,8 %",
            "bbox": {"x": 0.06, "y": 0.30, "w": 0.4, "h": 0.025},
            "cells": [],
            "geometric_group_id": "geometry_group_002",
        },
        {
            "row_id": "spatial_row_003",
            "line_id": "line_003",
            "text": "LUXUS TOILET. PAP",
            "bbox": {"x": 0.06, "y": 0.40, "w": 0.4, "h": 0.025},
            "cells": [],
            "geometric_group_id": "geometry_group_003",
        },
    ]


def test_region_candidates_include_quantity_support_and_clean_fallback_price() -> None:
    candidates = build_region_item_price_candidates(
        _visual_evidence(),
        spatial_rows=_spatial_rows(),
        page_width=1000,
        page_height=2000,
    )

    by_description = {candidate["description"]: candidate for candidate in candidates}
    assert by_description["KART. VORW. FESTK."]["line_total"] == 1.99
    assert by_description["H-MILCH 3,8 %"]["quantity"] == 2.0
    assert by_description["H-MILCH 3,8 %"]["unit_price"] == 1.29
    assert by_description["H-MILCH 3,8 %"]["line_total"] == 2.58
    assert by_description["LUXUS TOILET. PAP"]["line_total"] == 3.35
    assert (
        by_description["LUXUS TOILET. PAP"]["evidence_source"]
        == "region_clean_right_amount_closer_than_damaged_token"
    )
    # The damaged token is closer to WATTEPADS than the following clean 3.99,
    # therefore the next product price must not be stolen.
    assert "WATTEPADS MAXI" not in by_description
    # Product-like footer labels are not eligible for fallback pairing merely
    # because a total is horizontally aligned nearby.
    assert "AS-Zeit" not in by_description

    prompt = region_price_candidates_to_prompt_text(candidates)
    assert "SUPPLEMENTAL" not in prompt
    assert "region_line_total=3.35" in prompt
    assert "unit_price=1.29" in prompt


def test_line_price_repair_changes_only_matched_item_money_fields() -> None:
    candidates = build_region_item_price_candidates(
        _visual_evidence(),
        spatial_rows=_spatial_rows(),
        page_width=1000,
        page_height=2000,
    )
    receipt = {
        "merchant": {"name": "REWE"},
        "items": [
            {
                "description": "KART. VORW. FESTK.",
                "product_description": "KART. VORW. FESTK.",
                "category": "item",
                "line_total": 3.90,
                "source_line_ids": ["line_001", "line_010"],
                "confidence": 0.70,
            },
            {
                "description": "H-MILCH 3,8 %",
                "product_description": "H-MILCH 3,8 %",
                "category": "item",
                "quantity": 2.0,
                "unit_price": 0.64,
                "line_total": 1.29,
                "source_line_ids": ["line_002", "line_020"],
                "confidence": 0.80,
            },
            {
                "description": "LUXUS TOILET. PAP",
                "product_description": "LUXUS TOILET. PAP",
                "category": "item",
                "line_total": None,
                "source_line_ids": ["line_003"],
                "confidence": 0.60,
            },
            {
                "description": "UNRELATED PRODUCT",
                "product_description": "UNRELATED PRODUCT",
                "category": "item",
                "line_total": 9.99,
                "source_line_ids": ["line_004"],
                "confidence": 0.90,
            },
        ],
        "totals": {"grand_total": 17.91},
    }
    report = {
        "import_decision": "reject",
        "issues": [
            {"code": "ITEM_SUM_MISMATCH", "severity": "high"},
            {"code": "ITEMS_WITHOUT_LINE_TOTAL", "severity": "medium"},
        ],
    }
    document_map = {
        "rows": [
            {
                "line_id": "line_010",
                "text": "3,90",
                "confidence": 0.40,
                "bbox": {"x": 0.80, "y": 0.25, "w": 0.1, "h": 0.02},
                "cells": [{"amount_candidates": [{"value": 3.90}]}],
            },
            {
                "line_id": "line_020",
                "text": "1,29",
                "confidence": 0.95,
                "bbox": {"x": 0.40, "y": 0.35, "w": 0.1, "h": 0.02},
                "cells": [{"amount_candidates": [{"value": 1.29}]}],
            },
        ],
        "region_item_price_candidates": candidates,
    }

    repaired, actions = repair_receipt_line_prices(receipt, report, document_map)

    assert repaired["items"][0]["line_total"] == 1.99
    assert "line_010" not in repaired["items"][0]["source_line_ids"]
    assert repaired["items"][1]["line_total"] == 2.58
    assert repaired["items"][1]["quantity"] == 2.0
    assert repaired["items"][1]["unit_price"] == 1.29
    assert repaired["items"][2]["line_total"] == 3.35
    assert repaired["items"][3]["line_total"] == 9.99
    assert repaired["merchant"] == receipt["merchant"]
    assert repaired["totals"] == receipt["totals"]
    assert len(actions) == 3


def test_line_price_repair_updates_supported_quantity_metadata_without_replacing_total() -> None:
    receipt = {
        "items": [
            {
                "description": "BIO TK PETERSIL",
                "product_description": "BIO TK PETERSIL",
                "category": "item",
                "quantity": 2.0,
                "unit_price": 0.40,
                "line_total": 1.58,
                "source_line_ids": ["line_001"],
            }
        ]
    }
    report = {
        "issues": [{"code": "ITEM_SUM_MISMATCH", "severity": "high"}],
    }
    document_map = {
        "rows": [],
        "region_item_price_candidates": [
            {
                "candidate_id": "region_price_000",
                "description": "BIO TK PETERSIL",
                "line_total": 1.58,
                "quantity": 2.0,
                "unit_price": 0.79,
                "unit": "Stk",
                "layout_confidence": 0.98,
                "source_line_ids": ["region_line_001", "region_line_002"],
                "quantity_source_line_ids": ["region_line_003"],
                "evidence_source": "crop_ocr_right_amount_same_y_band",
            }
        ],
    }

    repaired, actions = repair_receipt_line_prices(receipt, report, document_map)

    assert repaired["items"][0]["line_total"] == 1.58
    assert repaired["items"][0]["quantity"] == 2.0
    assert repaired["items"][0]["unit_price"] == 0.79
    assert repaired["items"][0]["unit"] == "Stk"
    assert actions[0]["changed_fields"] == ["unit_price", "unit"]
