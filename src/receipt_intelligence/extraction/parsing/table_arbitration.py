#!/usr/bin/env python3
"""Table-evidence arbitration for OCR/VLM receipt rows.

This module does not create the final receipt. It builds a compact evidence
artifact that helps the LLM choose between conflicting sources:

- structured VLM table rows can be clean but sometimes row-shifted;
- OCR layout rows can preserve price/product pairing even when the VLM table is
  misaligned;
- quantity/unit-price rows explain adjacent product totals and should not be
  output as standalone products;
- percentages inside product names are not VAT/tax evidence without explicit
  tax context.

The output is evidence + warnings only. The LLM still performs semantic receipt
assembly; deterministic validation still decides import/readiness.
"""

from __future__ import annotations

import json
import re
from typing import Any

SCHEMA_VERSION = "v14_18_table_evidence_arbitration_1"

AMOUNT_TOL = 0.03
PRODUCT_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")
CONTEXT_RE = re.compile(
    r"\b(SUMME|TOTAL|GESAMT|ZWISCHENSUMME|SUBTOTAL|ZU\s*(?:ZAHLEN|BEZAHLEN)|BAR|BARGELD|CASH|EC|KARTE|CARD|GIROCARD|VISA|MASTERCARD|GUTSCHEIN|COUPON|RABATT|RÜCKG|RUECKG|CHANGE|DATUM|UHRZEIT|BELEG|KUNDENBELEG|TRACE|TERMINAL)\b",
    re.I,
)
TAX_CONTEXT_RE = re.compile(
    r"\b(MWST|M\.W\.ST|UST|U\.ST|VAT|TAX|STEUER|NETTO|BRUTTO|GROSS)\b", re.I
)
PERCENT_RE = re.compile(r"\b\d{1,2}(?:[,\.]\d+)?\s*%")
QTY_NOTE_LEFT_RE = re.compile(
    r"^\s*(\d+(?:[,\.]\d+)?)\s*(?:STK|STÜCK|STUECK|PCS?|QTY|ANZ|KG|G|GRAMM|L|ML|PACK|PK)?\s*(?:x|×|\*)\s*$",
    re.I,
)
QTY_NOTE_INLINE_RE = re.compile(
    r"\b(\d+(?:[,\.]\d+)?)\s*(?:STK|STÜCK|STUECK|PCS?|QTY|ANZ|KG|G|GRAMM|L|ML|PACK|PK)?\s*(?:x|×|\*)\s*([-+]?\d{1,5}(?:[,\.]\d{2}))\b",
    re.I,
)
TAX_CODE_SUFFIX_RE = re.compile(
    r"(?P<amount>[-+−]?\s*\d{1,5}(?:[.,]\d{2}|\s+\d{2}))\s*(?P<tax>[A-Za-z])\b"
)


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    s = str(value).strip().replace("−", "-")
    if not s:
        return None
    neg = s.startswith("-") or s.endswith("-")
    s = re.sub(r"[^0-9,.\s]", "", s).replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        v = round(float(s), 2)
    except Exception:
        return None
    return -abs(v) if neg else v


def _txt(value: Any) -> str:
    return str(value or "").strip()


def _is_product_like(text: Any) -> bool:
    t = _txt(text)
    if not t:
        return False
    if CONTEXT_RE.search(t):
        return False
    words = [
        w
        for w in PRODUCT_WORD_RE.findall(t)
        if not re.fullmatch(r"EUR|EURO|STK|PCS|QTY|ANZ|KG|GRAMM|ML|PACK", w, re.I)
    ]
    return bool(words)


def _is_quantity_note_row(row: dict[str, Any]) -> bool:
    left = _txt(row.get("left_text") or row.get("description") or row.get("product_description"))
    full = _txt(row.get("row_text") or row.get("full_text") or row.get("text"))
    return bool(QTY_NOTE_LEFT_RE.search(left) or QTY_NOTE_INLINE_RE.search(full))


def _parse_quantity_note(row: dict[str, Any]) -> dict[str, Any] | None:
    text = _txt(row.get("row_text") or row.get("full_text") or row.get("text"))
    m = QTY_NOTE_INLINE_RE.search(text)
    if m:
        qty = _num(m.group(1))
        unit = _num(m.group(2))
    else:
        left = _txt(row.get("left_text"))
        m = QTY_NOTE_LEFT_RE.search(left)
        qty = _num(m.group(1)) if m else None
        unit = _num(row.get("right_amount_value") or row.get("line_total") or row.get("unit_price"))
    if qty is None or unit is None:
        return None
    return {"quantity": qty, "unit_price": unit, "computed_total": round(qty * unit, 2)}


def _tax_code_from_raw(raw: Any) -> str | None:
    m = TAX_CODE_SUFFIX_RE.search(_txt(raw))
    if not m:
        return None
    return m.group("tax")


def build_ocr_layout_item_candidates(
    ocr_context: dict[str, Any], *, max_rows: int = 120
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    rows = [r for r in (ocr_context.get("layout_rows") or []) if isinstance(r, dict)]
    for row in rows[:max_rows]:
        val = _num(row.get("right_amount_value"))
        desc = _txt(row.get("left_text") or row.get("row_text") or row.get("text"))
        raw_amount = row.get("right_amount_raw")
        if val is None or not _is_product_like(desc):
            continue
        if _is_quantity_note_row(row):
            continue
        # Percent in product text is explicitly allowed as item evidence when there
        # are no tax keywords on the row.
        product_percent_not_tax = bool(PERCENT_RE.search(desc) and not TAX_CONTEXT_RE.search(desc))
        candidates.append(
            {
                "candidate_id": f"ocr_item_{len(candidates):03d}",
                "source": "ocr_layout_row",
                "row_id": row.get("row_id"),
                "source_line_ids": row.get("source_line_ids") or [],
                "description": desc,
                "line_total": val,
                "raw_amount": raw_amount,
                "tax_code": _tax_code_from_raw(raw_amount),
                "product_percent_not_tax": product_percent_not_tax,
                "evidence_text": _txt(
                    row.get("row_text") or row.get("full_text") or f"{desc} | {raw_amount}"
                ),
            }
        )
    return candidates


def build_quantity_note_candidates(
    ocr_context: dict[str, Any], item_candidates: list[dict[str, Any]], *, max_rows: int = 140
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows = [r for r in (ocr_context.get("layout_rows") or []) if isinstance(r, dict)]
    item_by_row = {str(c.get("row_id")): c for c in item_candidates}
    row_id_to_index = {str(r.get("row_id")): i for i, r in enumerate(rows)}
    for row in rows[:max_rows]:
        parsed = _parse_quantity_note(row)
        if not parsed:
            continue
        rid = str(row.get("row_id") or "")
        idx = row_id_to_index.get(rid, -1)
        matches: list[dict[str, Any]] = []
        for other in rows[max(0, idx - 4) : min(len(rows), idx + 5)] if idx >= 0 else []:
            oid = str(other.get("row_id") or "")
            cand = item_by_row.get(oid)
            if not cand:
                continue
            if (
                abs(float(cand.get("line_total") or 0.0) - float(parsed["computed_total"]))
                <= AMOUNT_TOL
            ):
                matches.append(
                    {
                        "row_id": cand.get("row_id"),
                        "description": cand.get("description"),
                        "line_total": cand.get("line_total"),
                    }
                )
        out.append(
            {
                "candidate_id": f"qty_note_{len(out):03d}",
                "source": "ocr_layout_row",
                "row_id": row.get("row_id"),
                "source_line_ids": row.get("source_line_ids") or [],
                "quantity": parsed["quantity"],
                "unit_price": parsed["unit_price"],
                "computed_total": parsed["computed_total"],
                "evidence_text": _txt(
                    row.get("row_text") or row.get("full_text") or row.get("text")
                ),
                "matching_item_rows": matches,
                "guidance": "This is quantity/unit-price evidence. Do not output it as a standalone item when it explains a nearby product line_total.",
            }
        )
    return out


def detect_vlm_table_alignment_warnings(
    visual_evidence: dict[str, Any], ocr_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    tables = [t for t in (visual_evidence.get("structured_tables") or []) if isinstance(t, dict)]
    if not tables or not ocr_items:
        return warnings
    for table in tables[:3]:
        rows = [r for r in (table.get("rows") or []) if isinstance(r, dict)]
        if len(rows) < 3:
            continue
        header = rows[0]
        header_text = " ".join(str(c) for c in (header.get("cells") or []))
        header_looks_like_product = _is_product_like(header_text) and not any(
            (_num(a.get("value")) is not None) for a in (header.get("amounts") or [])
        )
        first_data = next((r for r in rows[1:] if r.get("amounts")), None)
        if not first_data:
            continue
        first_vlm_amount = _num((first_data.get("amounts") or [{}])[0].get("value"))
        first_ocr = ocr_items[0]
        first_ocr_amount = _num(first_ocr.get("line_total"))
        first_vlm_desc = _txt(" ".join(str(c) for c in (first_data.get("cells") or [])[:1]))
        # Row-shift signature: a product-like header without amount, then the
        # first VLM data amount equals the first OCR product amount but the VLM
        # description is different from the OCR product description.
        if (
            header_looks_like_product
            and first_vlm_amount is not None
            and first_ocr_amount is not None
            and abs(first_vlm_amount - first_ocr_amount) <= AMOUNT_TOL
        ):
            if (
                first_vlm_desc
                and _txt(first_ocr.get("description"))
                and first_vlm_desc.upper()[:6] != _txt(first_ocr.get("description")).upper()[:6]
            ):
                warnings.append(
                    {
                        "code": "VLM_TABLE_POSSIBLE_ROW_SHIFT",
                        "severity": "medium",
                        "table_id": table.get("id"),
                        "reason": "The first VLM row was marked as header although it looks like an item without amount; the next VLM amount matches the first OCR layout item amount but with a different description.",
                        "vlm_header_text": header_text,
                        "first_vlm_data_text": first_data.get("row_text"),
                        "first_ocr_item": first_ocr,
                        "guidance": "Prefer OCR layout item-price pairing for this table unless VLM arithmetic clearly balances better.",
                    }
                )
    return warnings


def build_table_arbitration(
    visual_evidence: dict[str, Any] | None, ocr_context: dict[str, Any] | None
) -> dict[str, Any]:
    ve = visual_evidence if isinstance(visual_evidence, dict) else {}
    ctx = ocr_context if isinstance(ocr_context, dict) else {}
    ocr_items = build_ocr_layout_item_candidates(ctx)
    qty_notes = build_quantity_note_candidates(ctx, ocr_items)
    percent_rows = [c for c in ocr_items if c.get("product_percent_not_tax")]
    warnings = detect_vlm_table_alignment_warnings(ve, ocr_items)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if (ocr_items or warnings or qty_notes) else "empty",
        "summary": {
            "ocr_layout_item_candidate_count": len(ocr_items),
            "quantity_note_candidate_count": len(qty_notes),
            "product_percent_not_tax_count": len(percent_rows),
            "warning_count": len(warnings),
        },
        "warnings": warnings,
        "ocr_layout_item_candidates": ocr_items[:80],
        "quantity_note_candidates": qty_notes[:50],
        "product_percent_not_tax_rows": percent_rows[:30],
        "guidance": [
            "Use ocr_layout_item_candidates as cross-check evidence when structured VLM table rows appear shifted or misaligned.",
            "Do not output quantity_note_candidates as standalone items when they explain nearby product line totals.",
            "Rows listed in product_percent_not_tax_rows are product rows with percentage text, not tax rows unless explicit tax/MwSt/USt/VAT/Netto/Brutto context exists.",
        ],
    }


def attach_table_arbitration_to_visual_evidence(
    visual_evidence: dict[str, Any] | None, arbitration: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not isinstance(visual_evidence, dict):
        return visual_evidence
    if not isinstance(arbitration, dict):
        return visual_evidence
    enriched = dict(visual_evidence)
    enriched["table_arbitration"] = arbitration
    summary = dict(enriched.get("summary") or {})
    summary["table_arbitration_status"] = arbitration.get("status")
    summary["table_arbitration_warning_count"] = len(arbitration.get("warnings") or [])
    summary["ocr_layout_item_candidate_count"] = (arbitration.get("summary") or {}).get(
        "ocr_layout_item_candidate_count"
    )
    enriched["summary"] = summary
    guidance = list(enriched.get("semantic_guidance") or [])
    guidance.extend(arbitration.get("guidance") or [])
    enriched["semantic_guidance"] = guidance
    return enriched


def table_arbitration_to_prompt_text(arbitration: dict[str, Any] | None) -> str:
    if not isinstance(arbitration, dict):
        return ""
    return json.dumps(arbitration, ensure_ascii=False, indent=2)
