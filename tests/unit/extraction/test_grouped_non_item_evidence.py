from __future__ import annotations

from receipt_intelligence.extraction.evidence.grouped import (
    build_amount_only_product_attachment_candidates,
    build_grouped_evidence,
    build_semantic_row_context_candidates,
    grouped_evidence_to_prompt_text,
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


def test_semantic_rows_are_exposed_without_deterministic_item_classification() -> None:
    rows = [
        _row("layout_row_000", "ArtNr. Beschreibung Menge Summe", None),
        _row("layout_row_001", "23 Bella 1 7,50", 7.50),
        _row("layout_row_002", "*Norm*", None),
        _row("layout_row_003", "Summe 21,90", 21.90),
    ]

    candidates = build_semantic_row_context_candidates(rows)
    grouped = build_grouped_evidence(rows)
    prompt = grouped_evidence_to_prompt_text(grouped)

    assert len(candidates) == 4
    assert "semantic_row_context_candidates" in grouped
    assert "do_not_output_as_item_candidates" not in grouped
    assert "ROWS FOR RECEIPT-WIDE SEMANTIC CLASSIFICATION" in prompt
    assert "STRICT DO-NOT-OUTPUT-AS-ITEM ROWS" not in prompt


def test_summe_column_header_does_not_cut_off_amount_attachment_candidates() -> None:
    rows = [
        _row("layout_row_000", "ArtNr. Beschreibung Menge Summe", None),
        _row("layout_row_001", "Bella", None),
        _row("layout_row_002", "1", 7.50),
        _row("layout_row_003", "Rustika", None),
        _row("layout_row_004", "1", 7.50),
        _row("layout_row_005", "Summe", 15.00),
    ]

    candidates = build_amount_only_product_attachment_candidates(rows)
    descriptions = {candidate.get("description_candidate") for candidate in candidates}

    assert "Bella" in descriptions
    assert "Rustika" in descriptions
