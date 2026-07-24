from __future__ import annotations

from receipt_intelligence.extraction.evidence.grouped import (
    build_do_not_output_as_item_candidates,
)
from receipt_intelligence.extraction.evidence.layout import _tags_for_text


def _row(
    row_id: str,
    text: str,
    amount: float | None,
    *,
    tags: list[str] | None = None,
) -> dict:
    return {
        "row_id": row_id,
        "left_text": text,
        "full_text": text,
        "right_amount_raw": None if amount is None else str(amount),
        "right_amount_value": amount,
        "source_line_ids": [row_id.replace("layout_row", "line")],
        "hint_tags": tags or [],
    }


def test_bare_percentage_is_not_tagged_as_tax_keyword() -> None:
    tags = _tags_for_text("10% 1,40-", -1.40)

    assert "percentage_candidate" in tags
    assert "negative_amount" in tags
    assert "tax_keyword" not in tags


def test_discounts_and_product_named_gross_remain_available_to_semantic_parser() -> None:
    rows = [
        _row(
            "layout_row_000",
            "10% | 2,50-",
            -2.50,
            tags=["negative_amount", "percentage_candidate"],
        ),
        _row(
            "layout_row_001",
            "Tüten gross | 0,25",
            0.25,
            tags=["item_candidate"],
        ),
        _row(
            "layout_row_002",
            "*** Total | 47,80",
            47.80,
            tags=["total_keyword"],
        ),
        _row(
            "layout_row_003",
            "EC-Cash | 47,80",
            47.80,
            tags=["payment_keyword"],
        ),
    ]

    candidates = build_do_not_output_as_item_candidates(rows)
    evidence = [candidate["evidence_text"] for candidate in candidates]

    assert not any("10% | 2,50-" in value for value in evidence)
    assert not any("Tüten gross | 0,25" in value for value in evidence)
    assert any("*** Total | 47,80" in value for value in evidence)
    assert any("EC-Cash | 47,80" in value for value in evidence)
