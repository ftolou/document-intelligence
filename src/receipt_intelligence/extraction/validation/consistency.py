#!/usr/bin/env python3
"""Non-semantic receipt consistency post-processing.

This module normalizes dates, times, totals, taxes and payment arithmetic when
independent evidence supports the change. It never decides item identity, row
type, item order, note attachment or discount semantics; those belong to the
semantic LLM and are checked by deterministic validation.
"""

from __future__ import annotations

import copy
import re
from datetime import date
from typing import Any

from receipt_intelligence.extraction.evidence.grouped import build_grouped_evidence


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    s = str(value).strip().replace("−", "-")
    if not s:
        return None
    neg = s.startswith("-") or s.endswith("-")
    s = re.sub(r"[^0-9,\.\s]", "", s).replace(" ", "")
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


def _same(a: Any, b: Any, tol: float = 0.05) -> bool:
    x = _num(a)
    y = _num(b)
    return x is not None and y is not None and abs(x - y) <= tol


def _item_sum(items: list[dict[str, Any]]) -> float | None:
    vals = []
    for it in items or []:
        if isinstance(it, dict):
            v = _num(it.get("line_total"))
            if v is not None:
                vals.append(v)
    return round(sum(vals), 2) if vals else None


def _payment_amount(receipt: dict[str, Any]) -> float | None:
    vals = []
    for p in receipt.get("payments") or []:
        if isinstance(p, dict):
            v = _num(p.get("amount"))
            if v is not None:
                vals.append(v)
    if vals:
        return round(sum(vals), 2)
    totals = receipt.get("totals") if isinstance(receipt.get("totals"), dict) else {}
    return _num(totals.get("paid_total"))


def _visual_final_price_candidates(visual_evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(visual_evidence, dict):
        return []
    out = []
    for g in visual_evidence.get("final_price_adjustment_groups") or []:
        if isinstance(g, dict) and _num(g.get("final_sale_price_candidate")) is not None:
            out.append(g)
    return out


def _visual_tax_total_candidate(visual_evidence: dict[str, Any] | None) -> float | None:
    if not isinstance(visual_evidence, dict):
        return None
    t = visual_evidence.get("tax_table_candidates")
    if isinstance(t, dict):
        return _num(t.get("tax_total_candidate"))
    return None


def _visual_payment_reconciliation_candidates(
    visual_evidence: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(visual_evidence, dict):
        return []
    out = []
    for g in visual_evidence.get("total_payment_reconciliation_candidates") or []:
        if isinstance(g, dict) and _num(g.get("settlement_total")) is not None:
            out.append(g)
    return out


def _tokens(text: Any) -> set[str]:
    t = re.sub(r"[^A-ZÄÖÜẞ0-9]+", " ", str(text or "").upper())
    return {
        x
        for x in t.split()
        if len(x) >= 3
        and not re.fullmatch(r"(?:EUR|EURO|PREIS|IHR|RABATT|GRUND|STK|KG|MWST|NETTO|BRUTTO)", x)
    }


def _code_like_description(text: Any) -> bool:
    t = str(text or "")
    digits = sum(ch.isdigit() for ch in t)
    letters = sum(ch.isalpha() for ch in t)
    if re.search(r"\b(?:EAN|ART\.?NR|FAR\d{4,}|\d{10,14})\b", t, re.I):
        return True
    return digits >= 8 and digits >= letters


def _quantity_only_description(text: Any) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    words = re.findall(r"[A-Za-zÄÖÜäöüß]{3,}", t)
    non_unit = [
        w
        for w in words
        if not re.fullmatch(r"(?:STK|STUECK|STÜCK|PCS|KG|GRAMM|G|L|ML|EUR|EURO)", w, re.I)
    ]
    if non_unit:
        return False
    return bool(
        re.search(
            r"(?:^|\b)\d+[,.]?\d*\s*(?:KG|G|L|ML|STK|STÜCK|STUECK|PCS?)?\s*(?:x|×|@|à|a)?", t, re.I
        )
    )


def _region_final_price_groups(visual_evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract original->final price adjustment groups from region crop OCR rows.

    This is used only for consistency normalization. It is generic: an item row
    with an original/list price followed by an "Ihr Preis"/final-price row is
    represented as one final customer price. Discount/reason rows remain evidence,
    not standalone contributing items.
    """
    if not isinstance(visual_evidence, dict):
        return []
    blocks = visual_evidence.get("preferred_item_blocks") or []
    groups: list[dict[str, Any]] = []
    final_re = re.compile(
        r"\b(IHR\s+PREIS|DEIN\s+PREIS|AKTIONSPREIS|SALE\s*PRICE|ENDPREIS)\b", re.I
    )
    skip_re = re.compile(r"\b(RABATT|DISCOUNT|NACHLASS|GRUND|REASON|CODE|BON|SUMME)\b", re.I)
    for block in blocks:
        if not isinstance(block, dict):
            continue
        rows = [r for r in block.get("rows") or [] if isinstance(r, dict)]
        # Rows are already in crop reading order. Walk final price labels and pair
        # with the nearest preceding non-code product row that has an original price.
        used_prev: set[int] = set()
        for i, row in enumerate(rows):
            text = str(row.get("text") or row.get("description_candidate") or "")
            final = _num(row.get("amount"))
            if final is None or not final_re.search(text):
                continue
            best_idx = None
            for j in range(i - 1, -1, -1):
                if j in used_prev:
                    continue
                prev = rows[j]
                desc = str(prev.get("description_candidate") or prev.get("text") or "")
                orig = _num(prev.get("amount"))
                if orig is None or orig < final - 0.05:
                    continue
                if skip_re.search(desc) or _code_like_description(desc):
                    continue
                if not _tokens(desc):
                    continue
                best_idx = j
                break
            if best_idx is None:
                continue
            prev = rows[best_idx]
            used_prev.add(best_idx)
            orig = _num(prev.get("amount"))
            groups.append(
                {
                    "candidate_id": f"region_final_price_{len(groups):03d}",
                    "pattern": "region_original_price_final_price_group",
                    "product_row_id": prev.get("row_id"),
                    "product_row_text": prev.get("text") or prev.get("description_candidate"),
                    "product_description_candidate": prev.get("description_candidate")
                    or prev.get("text"),
                    "original_or_reference_price": orig,
                    "final_price_row_id": row.get("row_id"),
                    "final_price_row_text": text,
                    "final_sale_price_candidate": final,
                    "source": "region_crop_reocr",
                    "relationship_ok": final <= (orig or final) + 0.05,
                }
            )
    return groups


def _all_final_price_candidates(visual_evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    # Region-derived groups are more reliable for final-price grouping because
    # VLM table text can merge rows. Keep VLM groups as secondary evidence.
    return _region_final_price_groups(visual_evidence) + _visual_final_price_candidates(
        visual_evidence
    )


def _has_visible_total_or_payment_support(
    visual_evidence: dict[str, Any] | None, receipt: dict[str, Any]
) -> bool:
    if not isinstance(visual_evidence, dict):
        return False
    if _visual_payment_reconciliation_candidates(visual_evidence):
        return True
    best = visual_evidence.get("best_preferred_item_block")
    if (
        isinstance(best, dict)
        and isinstance(best.get("printed_total"), dict)
        and _num(best["printed_total"].get("amount")) is not None
    ):
        return True
    totals = receipt.get("totals") if isinstance(receipt.get("totals"), dict) else {}
    for sid in totals.get("source_line_ids") or []:
        if re.search(r"(?:total|summe|zahlen|bezahlen|bar|giro|ec|card|karte)", str(sid), re.I):
            return True
    return False


PRINTED_TOTAL_RE = re.compile(
    r"\b(?:SUMME|BONSUMME|BON\s*SUMME|GESAMT|GESAMTSUMME|TOTAL|ENDSUMME|ZU\s*(?:ZAHLEN|BEZAHLEN)|ZAHLBETRAG|AMOUNT\s*DUE)\b",
    re.IGNORECASE,
)
STRONG_PAYABLE_TOTAL_RE = re.compile(
    r"\b(?:BONSUMME|BON\s*SUMME|GESAMT|GESAMTSUMME|TOTAL|ENDSUMME|ZU\s*(?:ZAHLEN|BEZAHLEN)|ZAHLBETRAG|AMOUNT\s*DUE)\b",
    re.IGNORECASE,
)
GENERIC_SUMME_RE = re.compile(r"\bSUMME\b", re.IGNORECASE)
NON_FINAL_TOTAL_RE = re.compile(
    r"\b(?:ZWISCHENSUMME|SUBTOTAL|NETTO|NET|MWST|UST|VAT|TAX|STEUER|RABATT)\b", re.IGNORECASE
)
TAX_TABLE_CONTEXT_RE = re.compile(
    r"\b(?:MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|GROSS|INKL\.?\s*MWST)\b", re.IGNORECASE
)
INFO_ZERO_ITEM_RE = re.compile(
    r"\b(?:TERMIN|BESTELLUNG|BESTELLNR|ORDER|TIME|UHR|ABHOLUNG|LIEFERUNG|DELIVERY|PICKUP)\b",
    re.IGNORECASE,
)
DATE_TOKEN_RE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\b")
TIME_TOKEN_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")


def _normalize_date_token(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if m:
        y, mo, d = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        m = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b", text)
        if not m:
            return None
        d, mo, y = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if y < 100:
            # German retail receipts overwhelmingly use 20xx for current-era two-digit years.
            # Keep a conventional 1950 pivot for historical scans.
            y = 2000 + y if y <= 49 else 1900 + y
    try:
        return date(y, mo, d).isoformat()
    except Exception:
        return None


def _supported_date_candidates_from_context(ocr_context: dict[str, Any] | None) -> set[str]:
    if not isinstance(ocr_context, dict):
        return set()
    out: set[str] = set()
    containers = []
    containers.extend(ocr_context.get("date_time_candidates") or [])
    containers.extend(ocr_context.get("lines") or [])
    for row in containers:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("full_text") or "")
        for token in DATE_TOKEN_RE.findall(text):
            norm = _normalize_date_token(token)
            if norm:
                out.add(norm)
    return out


def _normalize_time_token(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    m = TIME_TOKEN_RE.search(text)
    if not m:
        return None
    h = int(m.group(1))
    minute = int(m.group(2))
    sec = int(m.group(3)) if m.group(3) is not None else None
    if not (0 <= h <= 23 and 0 <= minute <= 59):
        return None
    if sec is not None and not (0 <= sec <= 59):
        return None
    if sec is None:
        return f"{h:02d}:{minute:02d}"
    return f"{h:02d}:{minute:02d}:{sec:02d}"


def _supported_time_candidates_from_context(ocr_context: dict[str, Any] | None) -> set[str]:
    if not isinstance(ocr_context, dict):
        return set()
    out: set[str] = set()
    containers = []
    containers.extend(ocr_context.get("date_time_candidates") or [])
    containers.extend(ocr_context.get("lines") or [])
    for row in containers:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("full_text") or "")
        for m in TIME_TOKEN_RE.finditer(text):
            norm = _normalize_time_token(m.group(0))
            if norm:
                out.add(norm)
    return out


def _tax_table_evidence_ids_from_context(ocr_context: dict[str, Any] | None) -> set[str]:
    if not isinstance(ocr_context, dict):
        return set()
    try:
        ge = build_grouped_evidence(
            [r for r in (ocr_context.get("layout_rows") or []) if isinstance(r, dict)]
        )
    except Exception:
        return set()
    out: set[str] = set()
    for cand in ge.get("tax_table_candidates") or []:
        if not isinstance(cand, dict):
            continue
        for key in ("row_ids", "source_line_ids"):
            for sid in cand.get(key) or []:
                if sid is not None and str(sid).strip():
                    out.add(str(sid))
    return out


def _total_candidate_kind(text: str, ids: set[str], tax_table_ids: set[str]) -> tuple[str, int]:
    t = str(text or "")
    overlaps_tax_table = bool(ids.intersection(tax_table_ids)) or bool(
        TAX_TABLE_CONTEXT_RE.search(t)
    )
    if STRONG_PAYABLE_TOTAL_RE.search(t) and not re.search(
        r"\b(?:NETTO|NET|MWST|UST|VAT|TAX|STEUER)\b", t, re.I
    ):
        return "strong_payable_total", 100
    if GENERIC_SUMME_RE.search(t) and overlaps_tax_table:
        return "tax_table_summary_total", 10
    if PRINTED_TOTAL_RE.search(t) and not NON_FINAL_TOTAL_RE.search(t):
        return "generic_printed_total", 45
    return "non_final_or_weak_total", 0


def _printed_total_candidates_from_ocr_context(
    ocr_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(ocr_context, dict):
        return []
    out: list[dict[str, Any]] = []
    tax_table_ids = _tax_table_evidence_ids_from_context(ocr_context)
    for row in ocr_context.get("layout_rows") or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("full_text") or row.get("left_text") or "")
        tags = set(row.get("hint_tags") or [])
        val = _num(row.get("right_amount_value"))
        if val is None:
            continue
        source_ids = [str(x) for x in (row.get("source_line_ids") or []) if str(x).strip()]
        ids = {str(row.get("row_id") or "")} | set(source_ids)
        kind, strength = _total_candidate_kind(text, ids, tax_table_ids)
        if not (("total_keyword" in tags or PRINTED_TOTAL_RE.search(text)) and strength > 0):
            continue
        out.append(
            {
                "row_id": row.get("row_id"),
                "amount": val,
                "text": text,
                "source_line_ids": source_ids,
                "source": "ocr_layout_total_candidate",
                "total_kind": kind,
                "strength": strength,
            }
        )
    # Also use visible VLM/region printed totals when available in the OCR context registry.
    for line in ocr_context.get("visual_evidence_lines") or []:
        if not isinstance(line, dict):
            continue
        text = str(line.get("text") or "")
        source_ids = [str(line.get("line_id"))] if line.get("line_id") is not None else []
        ids = set(source_ids)
        kind, strength = _total_candidate_kind(text, ids, tax_table_ids)
        if not PRINTED_TOTAL_RE.search(text) or strength <= 0:
            continue
        for cand in line.get("amount_candidates") or []:
            val = _num(cand.get("value") if isinstance(cand, dict) else cand)
            if val is not None:
                out.append(
                    {
                        "row_id": line.get("line_id"),
                        "amount": val,
                        "text": text,
                        "source_line_ids": source_ids,
                        "source": "visual_evidence_total_candidate",
                        "total_kind": kind,
                        "strength": strength,
                    }
                )
    # Highest-confidence payable totals first; stable for equal candidates.
    out.sort(key=lambda c: int(c.get("strength") or 0), reverse=True)
    return out


def _remove_informational_zero_items(
    receipt: dict[str, Any], actions: list[dict[str, Any]], *, tolerance: float
) -> None:
    items = [it for it in (receipt.get("items") or []) if isinstance(it, dict)]
    if not items:
        return
    keep = []
    removed = []
    for it in items:
        desc = str(it.get("description") or "")
        v = _num(it.get("line_total"))
        if (
            str(it.get("category") or "").lower() == "item"
            and v is not None
            and abs(v) <= tolerance
            and INFO_ZERO_ITEM_RE.search(desc)
        ):
            removed.append(it)
        else:
            keep.append(it)
    if removed and keep:
        receipt["items"] = keep
        actions.append(
            {
                "action": "remove_informational_zero_value_items",
                "removed_count": len(removed),
                "removed": [
                    {
                        "description": it.get("description"),
                        "source_line_ids": it.get("source_line_ids"),
                    }
                    for it in removed[:12]
                ],
            }
        )


def _prefer_supported_printed_total(
    receipt: dict[str, Any],
    ocr_context: dict[str, Any] | None,
    actions: list[dict[str, Any]],
    *,
    tolerance: float,
) -> None:
    totals = receipt.setdefault("totals", {}) if isinstance(receipt.get("totals"), dict) else {}
    receipt["totals"] = totals
    candidates = _printed_total_candidates_from_ocr_context(ocr_context)
    if not candidates:
        return
    item_sum = _item_sum([it for it in (receipt.get("items") or []) if isinstance(it, dict)])
    subtotal = _num(totals.get("subtotal"))
    current_gt = _num(totals.get("grand_total"))
    paid_total = _num(totals.get("paid_total"))
    payment = _payment_amount(receipt)

    # If the current grand_total is already supported by payment/paid_total or a
    # strong payable label, never let a weak tax-table/generic Summe candidate
    # replace it merely because it equals subtotal/net. This fixes VAT-table
    # summaries such as Summe=Netto overriding zu zahlen/Kreditkarte totals.
    current_supported_by_payment = current_gt is not None and (
        (paid_total is not None and _same(current_gt, paid_total, tolerance))
        or (payment is not None and _same(current_gt, payment, tolerance))
    )
    current_supported_by_strong_total = current_gt is not None and any(
        _same(c.get("amount"), current_gt, tolerance) and int(c.get("strength") or 0) >= 100
        for c in candidates
    )

    best = None
    best_score = -1
    for cand in candidates:
        val = _num(cand.get("amount"))
        if val is None:
            continue
        strength = int(cand.get("strength") or 0)
        matches_item = item_sum is not None and abs(item_sum - val) <= max(tolerance, 0.05)
        matches_subtotal = subtotal is not None and abs(subtotal - val) <= max(tolerance, 0.05)
        matches_paid = (paid_total is not None and _same(paid_total, val, tolerance)) or (
            payment is not None and _same(payment, val, tolerance)
        )
        matches_current = current_gt is not None and _same(current_gt, val, tolerance)

        # Subtotal-only matches are weak: they are useful when no better total
        # anchor exists, but they must not override payable/payment evidence.
        if not (
            matches_item
            or matches_paid
            or matches_current
            or (
                matches_subtotal
                and not (current_supported_by_payment or current_supported_by_strong_total)
            )
        ):
            continue
        score = strength
        if matches_paid:
            score += 120
        if matches_item:
            score += 80
        if matches_current:
            score += 50
        if matches_subtotal:
            score += 10
        if score > best_score:
            best = cand
            best_score = score

    if best is None:
        return
    printed = _num(best.get("amount"))
    if printed is None:
        return
    if current_gt is not None and _same(current_gt, printed, tolerance):
        # Keep source IDs enriched even if no numeric change is required.
        src = list(totals.get("source_line_ids") or [])
        changed = False
        for sid in best.get("source_line_ids") or []:
            if sid not in src:
                src.append(sid)
                changed = True
        if changed:
            totals["source_line_ids"] = src
        return
    if current_gt is None or abs(current_gt - printed) > max(tolerance, 0.05):
        totals["grand_total"] = printed
        src = list(totals.get("source_line_ids") or [])
        for sid in best.get("source_line_ids") or []:
            if sid not in src:
                src.append(sid)
        totals["source_line_ids"] = src
        receipt["parse_status"] = (
            "ok" if receipt.get("parse_status") != "failed" else receipt.get("parse_status")
        )
        actions.append(
            {
                "action": "prefer_explicit_printed_total_supported_by_items_or_payment",
                "old_grand_total": current_gt,
                "new_grand_total": printed,
                "item_sum": item_sum,
                "subtotal": subtotal,
                "paid_total": paid_total,
                "payment": payment,
                "printed_total_evidence": best,
            }
        )


def _normalize_or_validate_date(
    receipt: dict[str, Any],
    ocr_context: dict[str, Any] | None,
    actions: list[dict[str, Any]],
) -> None:
    supported = _supported_date_candidates_from_context(ocr_context)
    if not supported:
        return
    current_raw = receipt.get("date")
    current = _normalize_date_token(current_raw)
    if current in supported:
        if current_raw != current:
            receipt["date"] = current
            actions.append(
                {
                    "action": "normalize_date_format_from_supported_evidence",
                    "old_date": current_raw,
                    "new_date": current,
                }
            )
        return
    # If the model hallucinated or mis-expanded a year and OCR has exactly one
    # distinct date, correct it deterministically instead of accepting a silent date error.
    if len(supported) == 1:
        new_date = next(iter(supported))
        if current_raw != new_date:
            receipt["date"] = new_date
            actions.append(
                {
                    "action": "correct_date_from_unique_ocr_date_evidence",
                    "old_date": current_raw,
                    "new_date": new_date,
                    "supported_dates": sorted(supported),
                }
            )


def _normalize_or_validate_time(
    receipt: dict[str, Any],
    ocr_context: dict[str, Any] | None,
    actions: list[dict[str, Any]],
) -> None:
    current_raw = receipt.get("time")
    if not current_raw:
        return
    current = _normalize_time_token(current_raw)
    supported = _supported_time_candidates_from_context(ocr_context)
    if current and (not supported or current in supported):
        if current_raw != current:
            receipt["time"] = current
            actions.append(
                {
                    "action": "normalize_time_format_from_supported_evidence",
                    "old_time": current_raw,
                    "new_time": current,
                }
            )
        return
    if len(supported) == 1:
        new_time = next(iter(supported))
        if current_raw != new_time:
            receipt["time"] = new_time
            actions.append(
                {
                    "action": "correct_time_from_unique_ocr_time_evidence",
                    "old_time": current_raw,
                    "new_time": new_time,
                    "supported_times": sorted(supported),
                }
            )
        return
    # Last safe fallback: if the model produced HH:MM:SS with invalid seconds,
    # strip seconds instead of preserving an impossible time like 14:10:86.
    m = TIME_TOKEN_RE.search(str(current_raw))
    if m:
        h = int(m.group(1))
        minute = int(m.group(2))
        sec = int(m.group(3)) if m.group(3) is not None else None
        if 0 <= h <= 23 and 0 <= minute <= 59 and sec is not None and sec > 59:
            new_time = f"{h:02d}:{minute:02d}"
            receipt["time"] = new_time
            actions.append(
                {
                    "action": "drop_invalid_time_seconds",
                    "old_time": current_raw,
                    "new_time": new_time,
                }
            )


def _remove_single_extra_unit_price_or_duplicate_context_item(
    receipt: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    tolerance: float,
) -> None:
    totals = receipt.get("totals") if isinstance(receipt.get("totals"), dict) else {}
    gt = _num(totals.get("grand_total"))
    items = [it for it in (receipt.get("items") or []) if isinstance(it, dict)]
    if gt is None or len(items) < 2:
        return
    old_sum = _item_sum(items)
    if old_sum is None or old_sum <= gt + max(tolerance, 0.05):
        return
    extra = round(old_sum - gt, 2)
    best_idx = None
    best_reason = None
    for idx, it in enumerate(items):
        v = _num(it.get("line_total"))
        if v is None or abs(v - extra) > max(tolerance, 0.05):
            continue
        desc = str(it.get("description") or "")
        ids = {str(x) for x in (it.get("source_line_ids") or []) if str(x).strip()}
        toks = _tokens(desc)
        has_overlap = False
        for j, other in enumerate(items):
            if j == idx:
                continue
            other_ids = {str(x) for x in (other.get("source_line_ids") or []) if str(x).strip()}
            other_toks = _tokens(other.get("description"))
            if ids and other_ids and ids.intersection(other_ids):
                has_overlap = True
                best_reason = "shared_source_line_with_another_item"
                break
            if toks and other_toks and len(toks.intersection(other_toks)) >= 1:
                has_overlap = True
                best_reason = "description_overlap_with_another_item"
                break
        context_like = bool(
            re.search(
                r"\b(?:PFAND|EXM|EINZEL|STK|STUECK|STÜCK|KG|G|L|ML|X|UNIT|EINZELPREIS)\b",
                desc,
                re.I,
            )
        )
        if has_overlap or context_like:
            candidate_items = [x for j, x in enumerate(items) if j != idx]
            new_sum = _item_sum(candidate_items)
            if new_sum is not None and abs(new_sum - gt) <= max(tolerance, 0.05):
                best_idx = idx
                if best_reason is None:
                    best_reason = "context_like_extra_amount_balances_grand_total"
                break
    if best_idx is None:
        return
    removed = items[best_idx]
    new_items = [x for j, x in enumerate(items) if j != best_idx]
    receipt["items"] = new_items
    actions.append(
        {
            "action": "remove_single_extra_unit_price_or_duplicate_context_item",
            "removed_description": removed.get("description"),
            "removed_line_total": removed.get("line_total"),
            "removed_source_line_ids": removed.get("source_line_ids"),
            "reason": best_reason,
            "old_item_sum": old_sum,
            "new_item_sum": _item_sum(new_items),
            "grand_total": gt,
        }
    )


def apply_consistency_postprocess(
    receipt: dict[str, Any],
    visual_evidence: dict[str, Any] | None = None,
    ocr_context: dict[str, Any] | None = None,
    *,
    tolerance: float = 0.05,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a copy with non-semantic, evidence-backed normalizations.

    This function may normalize dates, times, totals, tax arithmetic and payment
    reconciliation. It must never add, remove, reorder, rename or reclassify
    receipt items; semantic row interpretation belongs to the LLM.
    """
    r = copy.deepcopy(receipt) if isinstance(receipt, dict) else {}
    actions: list[dict[str, Any]] = []
    totals = r.setdefault("totals", {}) if isinstance(r.get("totals"), dict) else {}
    r["totals"] = totals

    _normalize_or_validate_date(r, ocr_context, actions)
    _normalize_or_validate_time(r, ocr_context, actions)

    # Row semantics remain owned by the LLM. Deterministic postprocessing must
    # not delete or relabel items from keyword or boundary heuristics.

    # If a printed Summe/Total row is visible and supported by item sum/subtotal,
    # prefer it over an invented subtotal+tax total.
    _prefer_supported_printed_total(r, ocr_context, actions, tolerance=tolerance)
    # Normalize tax rates like 0.19 -> 19.0 when the LLM encoded a rate as fraction.
    for tax in r.get("taxes") or []:
        if not isinstance(tax, dict):
            continue
        rate = _num(tax.get("rate"))
        if rate is not None and 0 < rate < 1:
            tax["rate"] = round(rate * 100.0, 2)
            actions.append(
                {"action": "normalize_tax_rate_fraction_to_percent", "new_rate": tax["rate"]}
            )

    # If taxes[].tax already exists, totals.tax_total should equal their sum.
    tax_vals = []
    for tax in r.get("taxes") or []:
        if isinstance(tax, dict):
            v = _num(tax.get("tax"))
            if v is not None:
                tax_vals.append(v)
    tax_sum = round(sum(tax_vals), 2) if tax_vals else None
    visual_tax_sum = _visual_tax_total_candidate(visual_evidence)
    preferred_tax_sum = tax_sum if tax_sum is not None else visual_tax_sum
    if preferred_tax_sum is not None:
        old_tax_total = _num(totals.get("tax_total"))
        if old_tax_total is None or abs(old_tax_total - preferred_tax_sum) > tolerance:
            # Stronger when old value equals a net/gross column or VLM tax candidate agrees.
            totals["tax_total"] = preferred_tax_sum
            actions.append(
                {
                    "action": "normalize_tax_total_from_extracted_tax_rows",
                    "old_tax_total": old_tax_total,
                    "new_tax_total": preferred_tax_sum,
                }
            )

    # Guard against impossible/suspicious tax_total values when no reliable tax
    # amount evidence exists. This is common when a noisy tax/net/gross table is
    # copied into totals.tax_total. For German/EU retail receipts, tax_total
    # should not be a large fraction of the gross total. If the model gives a
    # tax_total > 35% of grand_total and there is no extracted tax-row sum or
    # visual tax candidate, clear it instead of letting a net/gross value pollute
    # validation.
    gt_for_tax = _num(totals.get("grand_total"))
    old_tax_after = _num(totals.get("tax_total"))
    if (
        gt_for_tax is not None
        and old_tax_after is not None
        and old_tax_after > max(1.0, gt_for_tax * 0.35)
        and preferred_tax_sum is None
    ):
        totals["tax_total"] = None
        actions.append(
            {
                "action": "clear_suspicious_tax_total_without_reliable_tax_evidence",
                "old_tax_total": old_tax_after,
                "grand_total": gt_for_tax,
            }
        )

    # Payment-change reconciliation: if payment - abs(change) equals a printed final price/total candidate,
    # use it as grand_total instead of net/subtotal/tax rows.
    payment = _payment_amount(r)
    change = _num(totals.get("change"))
    settlement_total = (
        round(payment - abs(change), 2) if payment is not None and change is not None else None
    )
    final_candidates = _all_final_price_candidates(visual_evidence)
    recon_candidates = _visual_payment_reconciliation_candidates(visual_evidence)
    supported_settlement = False
    if settlement_total is not None:
        for c in recon_candidates:
            if _same(c.get("settlement_total"), settlement_total, tolerance):
                supported_settlement = True
                break
        for c in final_candidates:
            if _same(c.get("final_sale_price_candidate"), settlement_total, tolerance):
                supported_settlement = True
                break
    if settlement_total is not None and supported_settlement:
        old_gt = _num(totals.get("grand_total"))
        if old_gt is None or abs(old_gt - settlement_total) > tolerance:
            totals["grand_total"] = settlement_total
            actions.append(
                {
                    "action": "prefer_payment_change_supported_grand_total",
                    "old_grand_total": old_gt,
                    "new_grand_total": settlement_total,
                }
            )

    # Item identity, row type, note attachment, quantity interpretation and
    # discount semantics are intentionally not changed here. If validation
    # exposes an item-level inconsistency, the LLM correction pass receives the
    # complete visual context and proposes a validation-gated semantic rewrite.

    # Missing total/payment protection: if no reliable total/payment evidence
    # exists, do not let the model invent a grand_total from an incomplete item
    # section. Keep item evidence but mark the parse partial.
    printed_total_supports_gt = any(
        _same(c.get("amount"), totals.get("grand_total"), tolerance)
        for c in _printed_total_candidates_from_ocr_context(ocr_context)
    )
    if (
        _num(totals.get("grand_total")) is not None
        and _payment_amount(r) is None
        and not printed_total_supports_gt
        and not _has_visible_total_or_payment_support(visual_evidence, r)
    ):
        old_gt = _num(totals.get("grand_total"))
        totals["grand_total"] = None
        if totals.get("subtotal") == old_gt:
            totals["subtotal"] = None
        r["parse_status"] = "partial"
        warnings = list(r.get("warnings") or []) if isinstance(r.get("warnings"), list) else []
        warnings.append(
            "Grand total cleared because no reliable visible total/payment evidence was found; receipt appears partial or cropped."
        )
        r["warnings"] = warnings
        actions.append(
            {
                "action": "clear_unsupported_grand_total_without_total_payment_evidence",
                "old_grand_total": old_gt,
            }
        )

    return r, actions


def sanitize_model_warnings(
    receipt: dict[str, Any], validation_report: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drop model-generated warnings that contradict validator math after validation."""
    r = copy.deepcopy(receipt)
    actions = []
    warnings = r.get("warnings") if isinstance(r.get("warnings"), list) else []
    if (
        validation_report.get("balanced") is True
        and abs(float(validation_report.get("difference") or 0.0)) <= 0.05
    ):
        kept = []
        removed = []
        for w in warnings:
            txt = str(w)
            if re.search(
                r"(sum|gesamt|item|artikel).{0,80}(match|mismatch|does not match|stimmt nicht|abweich)",
                txt,
                re.I,
            ):
                removed.append(txt)
            elif re.search(
                r"(match|mismatch|does not match|stimmt nicht|abweich).{0,80}(sum|gesamt|item|artikel)",
                txt,
                re.I,
            ):
                removed.append(txt)
            else:
                kept.append(w)
        if removed:
            r["warnings"] = kept
            actions.append(
                {
                    "action": "remove_contradictory_model_arithmetic_warnings",
                    "removed_count": len(removed),
                    "removed_preview": removed[:3],
                }
            )
    return r, actions
