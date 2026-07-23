#!/usr/bin/env python3
"""
Compact receipt evidence builder with generic grouped evidence.

This module converts OCR/layout context into a small, receipt-shaped text
prompt for the LLM. It deliberately does NOT decide final receipt fields. It only
keeps the strongest layout rows and candidate hints while removing coordinates,
bboxes, source_word_ids, and bulky OCR JSON that confused local Gemma models.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from receipt_intelligence.extraction.evidence.grouped import (
    build_grouped_evidence,
    grouped_evidence_to_prompt_text,
)

DATE_RE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


KIND_PRIORITY = {
    "change_candidate": 0,
    "payment_candidate": 1,
    "discount_candidate": 2,
    "net_total_candidate_not_grand_total": 3,
    "tax_candidate": 4,
    "grand_total_candidate": 5,
    "discounted_item_price_candidate": 6,
    "quantity_price_row": 7,
    "quantity_unit_price_note": 8,
    "unpaired_amount_candidate": 9,
    "item_price_candidate": 10,
    "context_only": 11,
}


def _text(v: Any) -> str:
    return str(v or "").strip()


def _amount(v: Any) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def classify_layout_row(row: dict[str, Any]) -> str:
    """Classify evidence row for prompting only, not final parsing.

    Priority matters. Example: "IKEA FAMILY Rabatt Total" contains Total, but it
    is discount evidence, not grand total evidence.
    """
    tags = set(row.get("hint_tags") or [])
    value = row.get("right_amount_value")
    left = _text(row.get("left_text"))
    evidence_kind = _text(row.get("evidence_kind"))
    if value is None:
        if "quantity_unit_price_note" in tags:
            return "quantity_unit_price_note"
        return "context_only"
    if evidence_kind == "unpaired_amount_line":
        return "unpaired_amount_candidate"
    # IKEA-style quantity price blocks: a quantity such as 12,000 or 2,000 is
    # paired with unit/extended prices. This is not a standalone item name.
    if re.fullmatch(r"\d{1,4},\d{3}", left):
        return "quantity_price_row"
    if "change_keyword" in tags:
        return "change_candidate"
    if "payment_keyword" in tags:
        return "payment_candidate"
    if "discount_keyword" in tags or "negative_amount" in tags:
        return "discount_candidate"
    if "total_keyword" in tags and "net_keyword" in tags:
        return "net_total_candidate_not_grand_total"
    if "tax_keyword" in tags:
        return "tax_candidate"
    if "total_keyword" in tags:
        return "grand_total_candidate"
    if "price_override_keyword" in tags:
        return "discounted_item_price_candidate"
    if "quantity_unit_price_note" in tags:
        return "quantity_unit_price_note"
    return "item_price_candidate"


def _row_text(row: dict[str, Any]) -> str:
    left = _text(row.get("left_text"))
    raw = _text(row.get("right_amount_raw"))
    value = _amount(row.get("right_amount_value"))
    line_ids = ",".join(str(x) for x in (row.get("source_line_ids") or []))
    kind = classify_layout_row(row)
    if raw and value:
        pair = f"{left} | {raw} => {value}" if left else f"{raw} => {value}"
    elif raw:
        pair = f"{left} | {raw}" if left else raw
    else:
        pair = left or _text(row.get("full_text"))
    return f"[{row.get('row_id')}] {kind}: {pair}  (lines: {line_ids})"


def _has_signal(row: dict[str, Any]) -> bool:
    kind = classify_layout_row(row)
    if kind != "context_only":
        return True
    text = _text(row.get("full_text") or row.get("left_text"))
    return bool(DATE_RE.search(text) or TIME_RE.search(text))


def _select_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        return rows

    # Keep all semantic rows with values, date/time rows, and nearby context.
    keep_indexes: set[int] = set()
    for i, row in enumerate(rows):
        if _has_signal(row):
            keep_indexes.add(i)
            if i > 0:
                keep_indexes.add(i - 1)
            if i + 1 < len(rows):
                keep_indexes.add(i + 1)

    # Always keep first header rows and final footer rows.
    keep_indexes.update(range(min(12, len(rows))))
    keep_indexes.update(range(max(0, len(rows) - 18), len(rows)))

    selected = [rows[i] for i in sorted(keep_indexes) if 0 <= i < len(rows)]
    if len(selected) <= max_rows:
        return selected

    # If still too many, keep by semantic priority, then restore document order.
    scored = []
    for i, row in enumerate(selected):
        kind = classify_layout_row(row)
        priority = KIND_PRIORITY.get(kind, 50)
        scored.append((priority, i, row))
    kept = sorted(scored, key=lambda x: (x[0], x[1]))[:max_rows]
    kept_rows = [r for _, _, r in sorted(kept, key=lambda x: x[1])]
    return kept_rows


def _header_candidates(ocr_context: dict[str, Any], limit: int = 12) -> list[str]:
    lines = ocr_context.get("lines") or []
    out: list[str] = []
    for line in lines[:limit]:
        text = _text(line.get("text"))
        if not text:
            continue
        flags = set(line.get("flags") or [])
        if "has_amount" in flags and len(out) >= 4:
            continue
        out.append(f"[{line.get('line_id')}] {text}")
    return out


def _date_time_candidates(ocr_context: dict[str, Any], limit: int = 12) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for cand in ocr_context.get("date_time_candidates") or []:
        text = _text(cand.get("text"))
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(f"[{cand.get('line_id')}] {text}")
        if len(out) >= limit:
            break
    return out


def _candidate_summary(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        kind = classify_layout_row(row)
        if kind == "context_only":
            continue
        line = _row_text(row)
        buckets[kind].append(line)
    # Limit noisy item candidates but keep all important footer candidates.
    limits = {
        "item_price_candidate": 80,
        "discounted_item_price_candidate": 40,
        "quantity_unit_price_note": 40,
        "quantity_price_row": 60,
        "unpaired_amount_candidate": 30,
        "discount_candidate": 30,
        "grand_total_candidate": 20,
        "net_total_candidate_not_grand_total": 20,
        "payment_candidate": 20,
        "change_candidate": 20,
        "tax_candidate": 30,
    }
    return {
        k: v[: limits.get(k, 20)]
        for k, v in sorted(buckets.items(), key=lambda kv: KIND_PRIORITY.get(kv[0], 50))
    }


def build_compact_evidence(ocr_context: dict[str, Any], *, max_rows: int = 140) -> dict[str, Any]:
    rows = [r for r in (ocr_context.get("layout_rows") or []) if isinstance(r, dict)]
    selected_rows = _select_rows(rows, max_rows=max_rows)

    layout_lines = [_row_text(row) for row in selected_rows]
    grouped_evidence = build_grouped_evidence(rows)

    # Add only short before/after context for rows where it matters.
    neighbor_lines: list[str] = []
    for row in selected_rows:
        kind = classify_layout_row(row)
        if kind == "context_only":
            continue
        above = row.get("above_row") if isinstance(row.get("above_row"), dict) else None
        below = row.get("below_row") if isinstance(row.get("below_row"), dict) else None
        parts = []
        if above and _text(above.get("text")):
            parts.append(f"above={_text(above.get('text'))[:90]}")
        if below and _text(below.get("text")):
            parts.append(f"below={_text(below.get('text'))[:90]}")
        if parts:
            neighbor_lines.append(f"[{row.get('row_id')}] " + " ; ".join(parts))

    raw_lines = []
    # Minimal raw text: all selected source lines plus header/date/footer. No coordinates.
    selected_line_ids: set[str] = set()
    for row in selected_rows:
        selected_line_ids.update(str(x) for x in (row.get("source_line_ids") or []))
    all_lines = ocr_context.get("lines") or []
    for line in all_lines:
        lid = str(line.get("line_id") or "")
        txt = _text(line.get("text"))
        if not txt:
            continue
        if lid in selected_line_ids or len(raw_lines) < 10:
            raw_lines.append(f"[{lid}] {txt}")
    # Add bottom/footer lines because totals/payments are often there.
    for line in all_lines[-30:]:
        lid = str(line.get("line_id") or "")
        txt = _text(line.get("text"))
        entry = f"[{lid}] {txt}"
        if txt and entry not in raw_lines:
            raw_lines.append(entry)

    return {
        "schema_version": "v14_6_compact_evidence_1",
        "meta": {
            "line_count": ocr_context.get("line_count"),
            "kept_line_count": ocr_context.get("kept_line_count"),
            "layout_row_count": len(rows),
            "selected_layout_row_count": len(selected_rows),
            "omitted_middle_line_count": ocr_context.get("omitted_middle_line_count"),
        },
        "header_candidates": _header_candidates(ocr_context),
        "date_time_candidates": _date_time_candidates(ocr_context),
        "layout_rows_text": "\n".join(layout_lines),
        "candidate_summary": _candidate_summary(selected_rows),
        "grouped_evidence": grouped_evidence,
        "neighbor_context_text": "\n".join(neighbor_lines[:120]),
        "raw_text_minimal": "\n".join(raw_lines[:220]),
    }


def compact_evidence_to_prompt_text(evidence: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.append("EVIDENCE SUMMARY")
    meta = evidence.get("meta") or {}
    parts.append(
        f"line_count={meta.get('line_count')}, layout_rows={meta.get('layout_row_count')}, "
        f"selected_rows={meta.get('selected_layout_row_count')}, omitted_middle_lines={meta.get('omitted_middle_line_count')}"
    )
    if evidence.get("header_candidates"):
        parts.append("\nHEADER / MERCHANT CANDIDATES:\n" + "\n".join(evidence["header_candidates"]))
    if evidence.get("date_time_candidates"):
        parts.append("\nDATE / TIME CANDIDATES:\n" + "\n".join(evidence["date_time_candidates"]))
    if evidence.get("layout_rows_text"):
        parts.append("\nRECONSTRUCTED RECEIPT ROWS:\n" + evidence["layout_rows_text"])
    summary = evidence.get("candidate_summary") or {}
    if summary:
        chunks = []
        for kind, lines in summary.items():
            chunks.append(kind.upper() + ":\n" + "\n".join(lines))
        parts.append("\nBASIC CANDIDATE GROUPS:\n" + "\n\n".join(chunks))
    grouped_text = grouped_evidence_to_prompt_text(evidence.get("grouped_evidence") or {})
    if grouped_text:
        parts.append("\nGENERIC GROUPED EVIDENCE PATTERNS:\n" + grouped_text)
    if evidence.get("neighbor_context_text"):
        parts.append(
            "\nBEFORE / AFTER CONTEXT FOR AMOUNT ROWS:\n" + evidence["neighbor_context_text"]
        )
    if evidence.get("raw_text_minimal"):
        parts.append("\nMINIMAL RAW OCR TEXT, NO COORDINATES:\n" + evidence["raw_text_minimal"])
    return "\n".join(parts)
