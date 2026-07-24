from __future__ import annotations

from receipt_intelligence.extraction.evidence.spatial_document import (
    build_spatial_document_map,
    spatial_document_to_prompt_text,
)
from receipt_intelligence.extraction.parsing.llm_parser import build_ocr_context, build_prompt
from receipt_intelligence.extraction.parsing.spatial_overview import (
    build_geometry_only_overview,
    normalize_spatial_overview,
)


def _ocr_context() -> dict:
    raw = {
        "image_width": 1000,
        "image_height": 1600,
        "words": [
            {
                "id": "word_001",
                "text": "Pos",
                "confidence": 0.99,
                "xmin": 40,
                "ymin": 300,
                "xmax": 90,
                "ymax": 330,
            },
            {
                "id": "word_002",
                "text": "Artikel",
                "confidence": 0.99,
                "xmin": 130,
                "ymin": 300,
                "xmax": 260,
                "ymax": 330,
            },
            {
                "id": "word_003",
                "text": "Menge",
                "confidence": 0.99,
                "xmin": 590,
                "ymin": 300,
                "xmax": 680,
                "ymax": 330,
            },
            {
                "id": "word_004",
                "text": "Ges",
                "confidence": 0.99,
                "xmin": 880,
                "ymin": 300,
                "xmax": 940,
                "ymax": 330,
            },
            {
                "id": "word_005",
                "text": "2",
                "confidence": 0.99,
                "xmin": 50,
                "ymin": 360,
                "xmax": 65,
                "ymax": 390,
            },
            {
                "id": "word_006",
                "text": "21600276",
                "confidence": 0.99,
                "xmin": 130,
                "ymin": 360,
                "xmax": 260,
                "ymax": 390,
            },
            {
                "id": "word_007",
                "text": "1,00",
                "confidence": 0.99,
                "xmin": 600,
                "ymin": 360,
                "xmax": 660,
                "ymax": 390,
            },
            {
                "id": "word_008",
                "text": "1,79",
                "confidence": 0.99,
                "xmin": 880,
                "ymin": 360,
                "xmax": 940,
                "ymax": 390,
            },
        ],
    }
    return build_ocr_context(raw)


def test_spatial_document_preserves_words_columns_and_canvas() -> None:
    context = _ocr_context()
    document = build_spatial_document_map(context)

    assert context["words"]
    assert document["status"] == "ok"
    assert document["rows"][1]["cells"][0]["text"] == "2"
    assert document["rows"][1]["cells"][1]["text"] == "21600276"
    assert document["rows"][1]["cells"][2]["text"] == "1,00"
    assert document["rows"][1]["cells"][3]["text"] == "1,79"
    assert document["amount_column_candidates"]
    assert document["geometric_row_groups"]
    assert any(
        {cell["text"] for cell in group["cells"]} >= {"2", "21600276", "1,00", "1,79"}
        for group in document["geometric_row_groups"]
    )
    assert "21600276" in document["canvas"]
    assert "1,79" in spatial_document_to_prompt_text(document)


def test_spatial_main_prompt_makes_geometry_primary_not_table_authority() -> None:
    context = _ocr_context()
    document = build_spatial_document_map(context)
    prompt = build_prompt(
        context,
        spatial_document_map=document,
        spatial_overview={"schema_version": "spatial_overview_1", "status": "partial"},
        extraction_strategy="spatial_overview",
    )

    assert "GEOMETRIC ROW GROUPS" in prompt
    assert "Original OCR geometry and GEOMETRIC ROW GROUPS are primary evidence" in prompt
    assert "treat its item rows as AUTHORITATIVE" not in prompt
    assert "A position or article number must not become quantity" in prompt
    assert "STRICT DO-NOT-OUTPUT-AS-ITEM ROWS" not in prompt


def test_spatial_overview_drops_invented_line_ids() -> None:
    context = _ocr_context()
    document = build_spatial_document_map(context)
    valid_line_id = document["rows"][0]["line_id"]

    overview = normalize_spatial_overview(
        {
            "schema_version": "spatial_overview_1",
            "status": "ok",
            "sections": [
                {
                    "section_id": "items",
                    "type": "items",
                    "y_start": 0.1,
                    "y_end": 0.8,
                    "source_line_ids": [valid_line_id, "invented_line"],
                    "confidence": 0.9,
                }
            ],
            "tables": [],
            "line_annotations": [
                {
                    "line_id": "invented_line",
                    "row_type": "product",
                    "confidence": 1.0,
                }
            ],
            "warnings": [],
            "overall_confidence": 0.8,
        },
        document,
    )

    assert overview["sections"][0]["source_line_ids"] == [valid_line_id]
    assert overview["line_annotations"] == []


def test_geometry_only_overview_records_that_no_llm_call_was_made() -> None:
    document = build_spatial_document_map(_ocr_context())

    overview = build_geometry_only_overview(document)

    assert overview["status"] == "geometry_only"
    assert overview["mode"] == "deterministic_geometry"
    assert overview["llm_call_performed"] is False
    assert overview["geometric_row_group_count"] == len(
        document["geometric_row_groups"]
    )
    assert overview["prompt"] == ""
    assert overview["raw_output"] == ""
