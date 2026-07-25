"""Canonical receipt-wide spatial evidence for the receipt parser.

The existing OCR context preserves line boxes, but the compact main-parser
prompt intentionally flattens them.  This module keeps the original geometry
and derives only weak, auditable layout hints.  Semantic decisions remain the
responsibility of the receipt LLM.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from statistics import median
from typing import Any

from receipt_intelligence.extraction.evidence.layout import extract_ocr_amounts
from receipt_intelligence.extraction.evidence.region_price_fusion import (
    build_region_item_price_candidates,
    region_price_candidates_to_prompt_text,
)

JsonObject = dict[str, Any]

_TOTAL_RE = re.compile(
    r"\b(SUMME|TOTAL|GESAMT|BONSUMME|ENDSUMME|ZU\s+(?:ZAHLEN|BEZAHLEN)|AMOUNT\s+DUE)\b",
    re.IGNORECASE,
)
_TAX_RE = re.compile(r"\b(MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|GROSS|NET)\b", re.IGNORECASE)
_PAYMENT_RE = re.compile(
    r"\b(BAR|CASH|GEGEBEN|ZAHLUNG|KARTE|CARD|EC|GIROCARD|LASTSCHRIFT|VISA|MASTERCARD|R[ÜU]CKGELD|RUECKGELD|CHANGE)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
_TABLE_HEADER_RE = re.compile(
    r"\b(POS|POSITION|ARTIKEL|ARTICLE|MENGE|QTY|ANZAHL|STK|PREIS|PRICE|SUMME|TOTAL|EUR|NETTO|BRUTTO|MWST|VAT)\b",
    re.IGNORECASE,
)
_PRODUCT_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox(value: Any) -> JsonObject:
    raw = value if isinstance(value, dict) else {}
    return {
        "x": round(max(0.0, min(1.0, _float(raw.get("x")))), 5),
        "y": round(max(0.0, min(1.0, _float(raw.get("y")))), 5),
        "w": round(max(0.0, min(1.0, _float(raw.get("w")))), 5),
        "h": round(max(0.0, min(1.0, _float(raw.get("h")))), 5),
    }


def _bbox_union(boxes: Iterable[JsonObject]) -> JsonObject:
    values = [box for box in boxes if isinstance(box, dict)]
    if not values:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    x0 = min(_float(box.get("x")) for box in values)
    y0 = min(_float(box.get("y")) for box in values)
    x1 = max(_float(box.get("x")) + _float(box.get("w")) for box in values)
    y1 = max(_float(box.get("y")) + _float(box.get("h")) for box in values)
    return {
        "x": round(x0, 5),
        "y": round(y0, 5),
        "w": round(max(0.0, x1 - x0), 5),
        "h": round(max(0.0, y1 - y0), 5),
    }


def _line_hints(text: str, amount_count: int) -> list[str]:
    hints: list[str] = []
    if _DATE_RE.search(text):
        hints.append("date_or_time")
    if _TOTAL_RE.search(text):
        hints.append("total_or_subtotal")
    if _TAX_RE.search(text):
        hints.append("tax_or_net_gross")
    if _PAYMENT_RE.search(text):
        hints.append("payment_or_change")
    if _TABLE_HEADER_RE.search(text) and len(_TABLE_HEADER_RE.findall(text)) >= 2:
        hints.append("possible_table_header")
    if (
        amount_count
        and _PRODUCT_WORD_RE.search(text)
        and not any(
            hint in hints for hint in ("total_or_subtotal", "tax_or_net_gross", "payment_or_change")
        )
    ):
        hints.append("possible_item_row")
    if not hints:
        hints.append("unclassified")
    return hints


def _word_index(ocr_context: JsonObject) -> dict[str, JsonObject]:
    index: dict[str, JsonObject] = {}
    for raw in ocr_context.get("words") or []:
        if not isinstance(raw, dict):
            continue
        word_id = str(raw.get("word_id") or raw.get("id") or "").strip()
        text = str(raw.get("text") or "").strip()
        if not word_id or not text:
            continue
        index[word_id] = {
            "word_id": word_id,
            "text": text,
            "confidence": _float(raw.get("confidence")),
            "bbox": _bbox(raw.get("bbox")),
        }
    return index


def _group_words_into_cells(words: list[JsonObject]) -> list[JsonObject]:
    if not words:
        return []
    ordered = sorted(words, key=lambda row: _float((row.get("bbox") or {}).get("x")))
    cells: list[list[JsonObject]] = []
    for word in ordered:
        box = word.get("bbox") or {}
        x0 = _float(box.get("x"))
        if not cells:
            cells.append([word])
            continue
        previous = cells[-1][-1]
        prev_box = previous.get("bbox") or {}
        previous_end = _float(prev_box.get("x")) + _float(prev_box.get("w"))
        gap = x0 - previous_end
        # A visibly separated horizontal band becomes a cell.  This is only a
        # geometry grouping; no quantity/price/description role is assigned.
        if gap > 0.035:
            cells.append([word])
        else:
            cells[-1].append(word)

    result: list[JsonObject] = []
    for index, group in enumerate(cells):
        text = " ".join(str(word.get("text") or "") for word in group).strip()
        amounts = extract_ocr_amounts(text)
        result.append(
            {
                "cell_id": f"cell_{index:02d}",
                "text": text,
                "bbox": _bbox_union([word.get("bbox") or {} for word in group]),
                "word_ids": [str(word.get("word_id")) for word in group],
                "amount_candidates": amounts,
            }
        )
    return result


def _fallback_cells(line: JsonObject) -> list[JsonObject]:
    left = str(line.get("left_text") or "").strip()
    right = str(line.get("right_text") or "").strip()
    line_box = _bbox(line.get("bbox"))
    if left or right:
        cells: list[JsonObject] = []
        if left:
            cells.append(
                {
                    "cell_id": "cell_00",
                    "text": left,
                    "bbox": {
                        "x": line_box["x"],
                        "y": line_box["y"],
                        "w": round(line_box["w"] * 0.68, 5),
                        "h": line_box["h"],
                    },
                    "word_ids": [],
                    "amount_candidates": extract_ocr_amounts(left),
                }
            )
        if right:
            cells.append(
                {
                    "cell_id": f"cell_{len(cells):02d}",
                    "text": right,
                    "bbox": {
                        "x": round(line_box["x"] + line_box["w"] * 0.70, 5),
                        "y": line_box["y"],
                        "w": round(line_box["w"] * 0.30, 5),
                        "h": line_box["h"],
                    },
                    "word_ids": [],
                    "amount_candidates": extract_ocr_amounts(right),
                }
            )
        return cells
    text = str(line.get("text") or "").strip()
    return [
        {
            "cell_id": "cell_00",
            "text": text,
            "bbox": line_box,
            "word_ids": [],
            "amount_candidates": extract_ocr_amounts(text),
        }
    ]


def _cluster_amount_columns(rows: list[JsonObject]) -> list[JsonObject]:
    samples: list[float] = []
    for row in rows:
        for cell in row.get("cells") or []:
            if not isinstance(cell, dict) or not cell.get("amount_candidates"):
                continue
            box = cell.get("bbox") or {}
            samples.append(_float(box.get("x")) + _float(box.get("w")) / 2.0)
    clusters: list[list[float]] = []
    for sample in sorted(samples):
        for cluster in clusters:
            center = sum(cluster) / len(cluster)
            if abs(sample - center) <= 0.055:
                cluster.append(sample)
                break
        else:
            clusters.append([sample])
    return [
        {
            "column_candidate_id": f"amount_column_{index:02d}",
            "x_center": round(sum(cluster) / len(cluster), 5),
            "sample_count": len(cluster),
            "role": "unknown_amount_column",
        }
        for index, cluster in enumerate(clusters)
        if cluster
    ]


def _compact_visual_hypotheses(
    visual_evidence: JsonObject | None,
    arbitration: JsonObject | None,
) -> JsonObject:
    hypotheses: JsonObject = {}
    if isinstance(visual_evidence, dict):
        tables: list[JsonObject] = []
        for table in (visual_evidence.get("structured_tables") or [])[:4]:
            if not isinstance(table, dict):
                continue
            rows: list[JsonObject] = []
            for row in (table.get("rows") or [])[:40]:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "row_id": row.get("id"),
                        "cells": row.get("cells") or [],
                        "amounts": row.get("amounts") or [],
                    }
                )
            tables.append(
                {
                    "table_id": table.get("id"),
                    "headers": table.get("headers") or [],
                    "rows": rows,
                }
            )
        hypotheses["vlm"] = {
            "status": visual_evidence.get("status"),
            "summary": visual_evidence.get("summary") or {},
            "structured_tables": tables,
        }
    if isinstance(arbitration, dict):
        hypotheses["table_arbitration"] = {
            "summary": arbitration.get("summary") or {},
            "warnings": (arbitration.get("warnings") or [])[:8],
            "ocr_layout_item_candidates": (arbitration.get("ocr_layout_item_candidates") or [])[
                :80
            ],
        }
    return hypotheses


def _row_y_center(row: JsonObject) -> float:
    box = row.get("bbox") or {}
    return _float(box.get("y")) + _float(box.get("h")) / 2.0


def _build_geometric_row_groups(rows: list[JsonObject]) -> list[JsonObject]:
    """Cluster OCR lines that occupy the same printed horizontal band.

    The grouping uses normalized y geometry only. It does not assign product,
    discount, quantity, tax, or payment semantics. This gives the semantic LLM
    an auditable row overview without asking a separate model to rediscover
    alignment from flattened text.
    """

    if not rows:
        return []
    heights = [
        _float((row.get("bbox") or {}).get("h"))
        for row in rows
        if _float((row.get("bbox") or {}).get("h")) > 0
    ]
    median_height = median(heights) if heights else 0.02
    center_tolerance = max(0.0035, min(0.015, median_height * 0.42))

    raw_groups: list[list[JsonObject]] = []
    group_centers: list[float] = []
    for row in sorted(rows, key=lambda value: (_row_y_center(value), value.get("page_order", 0))):
        center = _row_y_center(row)
        if raw_groups and abs(center - group_centers[-1]) <= center_tolerance:
            raw_groups[-1].append(row)
            group_centers[-1] = sum(_row_y_center(item) for item in raw_groups[-1]) / len(
                raw_groups[-1]
            )
        else:
            raw_groups.append([row])
            group_centers.append(center)

    groups: list[JsonObject] = []
    for index, group_rows in enumerate(raw_groups):
        group_id = f"geometry_group_{index:03d}"
        cells: list[JsonObject] = []
        source_line_ids: list[str] = []
        for row in group_rows:
            line_id = str(row.get("line_id") or "")
            if line_id and line_id not in source_line_ids:
                source_line_ids.append(line_id)
            row["geometric_group_id"] = group_id
            for cell in row.get("cells") or []:
                if not isinstance(cell, dict):
                    continue
                cells.append(
                    {
                        "cell_id": f"{line_id}:{cell.get('cell_id')}",
                        "line_id": line_id,
                        "text": cell.get("text"),
                        "bbox": cell.get("bbox") or {},
                        "word_ids": cell.get("word_ids") or [],
                        "amount_candidates": cell.get("amount_candidates") or [],
                    }
                )
        cells.sort(
            key=lambda cell: (
                _float((cell.get("bbox") or {}).get("x")),
                str(cell.get("line_id") or ""),
            )
        )
        groups.append(
            {
                "group_id": group_id,
                "page_order": index,
                "y_center": round(group_centers[index], 5),
                "bbox": _bbox_union([row.get("bbox") or {} for row in group_rows]),
                "source_line_ids": source_line_ids,
                "cells": cells,
            }
        )
    return groups


def _geometric_groups_to_prompt_text(groups: list[JsonObject], *, max_groups: int) -> str:
    lines: list[str] = []
    for group in groups[:max_groups]:
        source_ids = ",".join(str(value) for value in group.get("source_line_ids") or [])
        lines.append(
            f"[{group.get('group_id')} y={_float(group.get('y_center')):.4f} lines={source_ids}]"
        )
        for cell in group.get("cells") or []:
            if not isinstance(cell, dict):
                continue
            box = cell.get("bbox") or {}
            x0 = _float(box.get("x"))
            x1 = x0 + _float(box.get("w"))
            amount_values = [
                candidate.get("value")
                for candidate in cell.get("amount_candidates") or []
                if isinstance(candidate, dict) and candidate.get("value") is not None
            ]
            amount_suffix = f" amounts={amount_values}" if amount_values else ""
            lines.append(
                f"  x={x0:.4f}..{x1:.4f} {cell.get('line_id')}: "
                f"{str(cell.get('text') or '').strip()}{amount_suffix}"
            )
    return "\n".join(lines)


def _hypotheses_to_prompt_text(hypotheses: JsonObject) -> str:
    """Render only compact secondary hypotheses; geometry stays authoritative."""

    vlm = hypotheses.get("vlm") if isinstance(hypotheses, dict) else None
    if not isinstance(vlm, dict):
        return "none"
    compact = {
        "status": vlm.get("status"),
        "structured_tables": vlm.get("structured_tables") or [],
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def build_spatial_canvas(rows: list[JsonObject], *, width: int = 112) -> str:
    """Render normalized x positions into a compact monospace page overview."""
    width = max(72, min(160, int(width)))
    header = (
        " " * 13
        + "0.0"
        + " " * max(1, width // 2 - 6)
        + "0.5"
        + " " * max(1, width // 2 - 6)
        + "1.0"
    )
    output = [header[: width + 13]]
    for row in rows:
        line_id = str(row.get("line_id") or row.get("row_id") or "line")
        canvas = [" "] * width
        cells = row.get("cells") or []
        if not cells:
            cells = [{"text": row.get("text"), "bbox": row.get("bbox")}]
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            text = re.sub(r"\s+", " ", str(cell.get("text") or "")).strip()
            if not text:
                continue
            box = cell.get("bbox") or {}
            start = min(width - 1, max(0, int(round(_float(box.get("x")) * (width - 1)))))
            available = max(1, width - start)
            text = text[:available]
            for offset, character in enumerate(text):
                pos = start + offset
                if pos >= width:
                    break
                if canvas[pos] != " " and character != " ":
                    # Keep both pieces visible instead of silently overwriting.
                    next_space = next((i for i in range(pos, width) if canvas[i] == " "), None)
                    if next_space is None:
                        break
                    pos = next_space
                canvas[pos] = character
        output.append(f"[{line_id:<10}] {''.join(canvas).rstrip()}")
    return "\n".join(output)


def build_spatial_document_map(
    ocr_context: JsonObject,
    *,
    visual_evidence: JsonObject | None = None,
    arbitration: JsonObject | None = None,
    canvas_width: int = 112,
) -> JsonObject:
    """Build the canonical spatial artifact from OCR geometry and hypotheses."""
    words = _word_index(ocr_context)
    rows: list[JsonObject] = []
    for index, raw_line in enumerate(ocr_context.get("lines") or []):
        if not isinstance(raw_line, dict):
            continue
        line_id = str(raw_line.get("line_id") or f"line_{index:03d}")
        source_word_ids = [str(value) for value in raw_line.get("source_word_ids") or []]
        line_words = [words[word_id] for word_id in source_word_ids if word_id in words]
        cells = _group_words_into_cells(line_words) or _fallback_cells(raw_line)
        text = str(raw_line.get("text") or "").strip()
        amount_count = sum(len(cell.get("amount_candidates") or []) for cell in cells)
        rows.append(
            {
                "row_id": f"spatial_row_{index:03d}",
                "line_id": line_id,
                "text": text,
                "bbox": _bbox(raw_line.get("bbox")),
                "confidence": _float(raw_line.get("confidence")),
                "source_word_ids": source_word_ids,
                "hints": _line_hints(text, amount_count),
                "cells": cells,
            }
        )
    rows.sort(
        key=lambda row: (
            _float((row.get("bbox") or {}).get("y")),
            _float((row.get("bbox") or {}).get("x")),
        )
    )
    for index, row in enumerate(rows):
        row["page_order"] = index

    geometric_row_groups = _build_geometric_row_groups(rows)
    region_item_price_candidates = build_region_item_price_candidates(
        visual_evidence,
        spatial_rows=rows,
        page_width=int(ocr_context.get("image_width") or 0),
        page_height=int(ocr_context.get("image_height") or 0),
    )
    canvas = build_spatial_canvas(rows, width=canvas_width)
    return {
        "schema_version": "spatial_document_map_1",
        "status": "ok" if rows else "empty",
        "page": {
            "width": int(ocr_context.get("image_width") or 0),
            "height": int(ocr_context.get("image_height") or 0),
        },
        "word_count": len(words),
        "line_count": len(rows),
        "rows": rows,
        "geometric_row_group_count": len(geometric_row_groups),
        "geometric_row_groups": geometric_row_groups,
        "amount_column_candidates": _cluster_amount_columns(rows),
        "region_item_price_candidate_count": len(region_item_price_candidates),
        "region_item_price_candidates": region_item_price_candidates,
        "hypotheses": _compact_visual_hypotheses(
            visual_evidence,
            arbitration,
        ),
        "evidence_policy": {
            "primary": "full_image_ocr_geometry",
            "supplemental_high_resolution": "region_crop_ocr_product_price_candidates",
            "secondary": ["vlm_tables", "table_arbitration"],
            "rule": (
                "Full-image geometry preserves page structure. High-confidence region crop OCR "
                "may correct a damaged or missing line price only when product text, spatial "
                "alignment, and explicit region source lines agree. VLM tables remain fallible "
                "hypotheses."
            ),
        },
        "canvas": canvas,
    }


def spatial_document_to_prompt_text(document_map: JsonObject, *, max_rows: int = 180) -> str:
    """Render the compact geometry-first evidence used by the main parser.

    The full document map remains available as an artifact. The prompt receives
    one canvas, geometric row groups, amount-column centers, and compact raw VLM
    table hypotheses. It intentionally omits duplicate flattened evidence.
    """

    groups = [
        group
        for group in (document_map.get("geometric_row_groups") or [])
        if isinstance(group, dict)
    ]
    return (
        "SPATIAL CANVAS (horizontal alignment reflects normalized OCR x positions):\n"
        + str(document_map.get("canvas") or "")
        + "\n\nGEOMETRIC ROW GROUPS (same-band clustering only; no semantic labels):\n"
        + _geometric_groups_to_prompt_text(groups, max_groups=max_rows)
        + "\n\nSUPPLEMENTAL REGION ITEM-PRICE CANDIDATES "
        "(high-resolution crop OCR; explicit source lines; do not invent rows):\n"
        + region_price_candidates_to_prompt_text(
            [
                candidate
                for candidate in (document_map.get("region_item_price_candidates") or [])
                if isinstance(candidate, dict)
            ],
            limit=max_rows,
        )
        + "\n\nAMOUNT COLUMN CENTERS:\n"
        + json.dumps(
            document_map.get("amount_column_candidates") or [],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\nSECONDARY VLM TABLE HYPOTHESES (fallible):\n"
        + _hypotheses_to_prompt_text(document_map.get("hypotheses") or {})
    )


__all__ = [
    "build_spatial_canvas",
    "build_spatial_document_map",
    "spatial_document_to_prompt_text",
]
