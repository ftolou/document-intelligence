#!/usr/bin/env python3
"""
Receipt layout-context builder.

This module prepares receipt-shaped OCR evidence for the LLM. It deliberately
DOES NOT produce final semantic receipt fields such as items, totals, taxes, or
payments. It only groups spatially-near OCR lines into aligned rows and produces
candidate/hint evidence that the LLM can accept or reject.
"""

from __future__ import annotations

import re
from statistics import median
from typing import Any

# OCR money candidates: German receipt amounts should normally use decimal comma.
# We intentionally do not treat 09.12 as money because it is often a date.
OCR_MONEY_RE = re.compile(
    r"(?<![\d/])("
    r"[-+−]?\s*\d{1,5}(?:[.\s]\d{3})*,\s*\d{2}"  # decimal comma / thousands dot
    r"|[-+−]?\s*\d{1,5}\.\d{2}"  # OCR dot decimal, e.g. 0.75
    r"|[-+−]?\s*\d{1,5}\s+\d{2}"  # OCR split decimal, e.g. 12 34
    r")(?:\s*[-−])?(?![\d/])"
)

DATE_RE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

CURRENCY_ONLY_RE = re.compile(r"^\s*(EUR|EURO|€)\s*$", re.IGNORECASE)
TOTAL_RE = re.compile(
    r"\b(SUMME|BON\s*SUMME|BONSUMME|GESAMT|TOTAL|ENDSUMME|ZU\s*ZAHLEN|BETRAG)\b", re.IGNORECASE
)
NET_RE = re.compile(r"\b(NETTO|OHNE\s+MWST|OHNE\s+UST)\b", re.IGNORECASE)
TAX_RE = re.compile(
    r"\b(MWST|M\.?W\.?ST|UST|U\.?ST|MEHRWERTST|STEUER|VAT|TAX)\b",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"(?<!\d)\d{1,2}(?:[,.]\d)?\s*%")
PAYMENT_RE = re.compile(
    r"\b(BAR|CASH|GEGEBEN|ZAHLUNG|KARTENZAHLUNG|LASTSCHRIFT|EC|GIROCARD|KARTE|VISA|MASTERCARD|MAESTRO|PAYPAL|KREDITKARTE|DEBIT)\b",
    re.IGNORECASE,
)
CHANGE_RE = re.compile(r"\b(R[ÜUO]CKGELD|RUECKGELD|ROCKGELD|WECHSELGELD|CHANGE)\b", re.IGNORECASE)
DISCOUNT_RE = re.compile(
    r"\b(RABATT|AKTION|AKTIONSRABATT|COUPON|BONUS|GUTSCHRIFT|NACHLASS)\b", re.IGNORECASE
)
PRICE_OVERRIDE_RE = re.compile(
    r"\b(IHR\s+PREIS|DEIN\s+PREIS|AKTIONSPREIS|ENDPREIS|PREIS)\b", re.IGNORECASE
)
UNIT_EXPR_RE = re.compile(r"\b\d+(?:[,.]\d+)?\s*[x×*]\s*\d{1,5},\d{2}\b", re.IGNORECASE)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _amount_from_raw(raw: str) -> float | None:
    s0 = str(raw or "").strip().replace("−", "-")
    if not s0:
        return None
    negative = (
        s0.startswith("-") or s0.endswith("-") or re.search(r",\s*\d{2}\s*[-−]", s0) is not None
    )
    s = s0.replace("−", "-")
    s = re.sub(r"[^0-9,\.\s+-]", "", s).strip()
    s = s.replace("-", "").replace("+", "").replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "." in s:
        # Dot-decimal OCR amounts are allowed only after date/time guards in
        # extract_ocr_amounts(); this recovers prices like "0.75" without
        # turning "09.12.17" into money.
        pass
    else:
        return None
    try:
        val = round(float(s), 2)
    except Exception:
        return None
    return -abs(val) if negative else val


def extract_ocr_amounts(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    if DATE_RE.search(text) or TIME_RE.search(text):
        # Date/time lines can contain 09.12; do not emit money candidates from them.
        # If they also contain a real comma amount, the regex below can still find it
        # after this guard is relaxed by the caller in the future.
        if "," not in text:
            return []
    out: list[dict[str, Any]] = []
    for match in OCR_MONEY_RE.finditer(text):
        raw = match.group(0).strip()
        value = _amount_from_raw(raw)
        if value is None:
            continue
        out.append({"raw": raw, "value": value, "span": [match.start(), match.end()]})
    return out


def _line_center(line: dict[str, Any]) -> tuple[float, float]:
    b = line.get("bbox") if isinstance(line.get("bbox"), dict) else {}
    x = _num(b.get("x")) + _num(b.get("w")) / 2.0
    y = _num(b.get("y")) + _num(b.get("h")) / 2.0
    return x, y


def _bbox_part(line: dict[str, Any], key: str) -> float:
    b = line.get("bbox") if isinstance(line.get("bbox"), dict) else {}
    return _num(b.get(key))


def _is_amount_only(text: str, amounts: list[dict[str, Any]]) -> bool:
    if not amounts:
        return False
    t = str(text or "").strip()
    remainder = t
    for a in amounts:
        remainder = remainder.replace(str(a.get("raw") or ""), " ")
    remainder = re.sub(r"[\s.,:;*x×+\-−]", "", remainder, flags=re.IGNORECASE)
    # E/V/A/B can be German tax markers after item prices; EUR is currency.
    return remainder.upper() in {"", "E", "V", "A", "B", "EUR", "EURO", "€"}


def _text_without_amounts(text: str, amounts: list[dict[str, Any]]) -> str:
    out = str(text or "")
    for a in amounts:
        out = out.replace(str(a.get("raw") or ""), " ")
    out = re.sub(r"\s+", " ", out).strip(" :;|-")
    return out


def _tags_for_text(text: str, amount_value: float | None = None) -> list[str]:
    t = str(text or "")
    tags: list[str] = []
    if TOTAL_RE.search(t):
        tags.append("total_keyword")
    if NET_RE.search(t):
        tags.append("net_keyword")
    if TAX_RE.search(t):
        tags.append("tax_keyword")
    if PERCENT_RE.search(t):
        tags.append("percentage_candidate")
    if PAYMENT_RE.search(t):
        tags.append("payment_keyword")
    if CHANGE_RE.search(t):
        tags.append("change_keyword")
    if DISCOUNT_RE.search(t):
        tags.append("discount_keyword")
    if PRICE_OVERRIDE_RE.search(t):
        tags.append("price_override_keyword")
    if UNIT_EXPR_RE.search(t):
        tags.append("quantity_unit_price_note")
    if DATE_RE.search(t):
        tags.append("date_candidate")
    if TIME_RE.search(t):
        tags.append("time_candidate")
    if CURRENCY_ONLY_RE.fullmatch(t.strip()):
        tags.append("currency_only")
    if amount_value is not None and amount_value < 0:
        tags.append("negative_amount")
    return tags


def _label_score(label: dict[str, Any], amount_line: dict[str, Any], row_threshold: float) -> float:
    _, ly = _line_center(label)
    _, ay = _line_center(amount_line)
    t = str(label.get("text") or "")
    dist = abs(ly - ay)
    # Distance must dominate. Keyword bonuses caused product prices to be paired
    # with lower footer/discount labels. Keywords are useful only as tie-breakers.
    score = dist
    if CURRENCY_ONLY_RE.fullmatch(t.strip()):
        score += row_threshold * 8.0
    has_keyword = (
        TOTAL_RE.search(t)
        or PAYMENT_RE.search(t)
        or CHANGE_RE.search(t)
        or TAX_RE.search(t)
        or DISCOUNT_RE.search(t)
    )
    if len(t.strip()) <= 3 and not has_keyword:
        score += row_threshold * 1.5
    if has_keyword:
        score -= row_threshold * 0.10
    return score


def build_layout_context(
    lines: list[dict[str, Any]], *, image_width: int = 1, image_height: int = 1
) -> dict[str, Any]:
    """Build receipt-shaped layout evidence from compact OCR lines.

    The returned data is for prompt evidence only. It should not be imported as
    final receipt data without LLM interpretation + validation.
    """
    enriched: list[dict[str, Any]] = []
    for i, line in enumerate(lines or []):
        if not isinstance(line, dict):
            continue
        text = str(line.get("text") or "").strip()
        amounts = extract_ocr_amounts(text)
        # Preserve precomputed amount candidates only when they do not come from dot dates.
        if (
            not amounts
            and isinstance(line.get("amount_candidates"), list)
            and not (DATE_RE.search(text) or TIME_RE.search(text))
        ):
            for cand in line.get("amount_candidates") or []:
                raw = str(cand.get("raw") or "")
                value = _amount_from_raw(raw) if "," in raw else None
                if value is not None:
                    amounts.append({"raw": raw, "value": value, "span": cand.get("span") or []})
        x, y = _line_center(line)
        h = max(_bbox_part(line, "h"), 0.0001)
        amount_value = amounts[-1]["value"] if amounts else None
        e = dict(line)
        e["amount_candidates"] = amounts
        e["x_center"] = round(x, 4)
        e["y_center"] = round(y, 4)
        e["height_norm"] = round(h, 4)
        e["amount_only"] = _is_amount_only(text, amounts)
        e["text_without_amounts"] = _text_without_amounts(text, amounts)
        e["layout_tags"] = _tags_for_text(text, amount_value)
        enriched.append(e)

    enriched.sort(key=lambda r: (_line_center(r)[1], _line_center(r)[0]))
    heights = [max(_bbox_part(l, "h"), 0.0001) for l in enriched]
    med_h = median(heights) if heights else 0.025
    row_threshold = max(0.010, min(0.035, med_h * 0.85))

    used_label_ids: set[str] = set()
    used_amount_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    amount_lines = [l for l in enriched if l.get("amount_candidates")]
    label_lines = [
        l
        for l in enriched
        if not l.get("amount_only")
        and not CURRENCY_ONLY_RE.fullmatch(str(l.get("text") or "").strip())
    ]

    for amount_line in amount_lines:
        lid = str(amount_line.get("line_id") or "")
        if not amount_line.get("amount_only"):
            # Mixed text+amount line, e.g. "Ihr Preis: 47,45".
            a = (amount_line.get("amount_candidates") or [])[-1]
            left = _text_without_amounts(
                str(amount_line.get("text") or ""), amount_line.get("amount_candidates") or []
            )
            rows.append(
                {
                    "row_id": f"layout_row_{len(rows):03d}",
                    "evidence_kind": "same_line_text_amount",
                    "left_text": left or str(amount_line.get("text") or ""),
                    "middle_text": "",
                    "right_amount_raw": a.get("raw"),
                    "right_amount_value": a.get("value"),
                    "full_text": str(amount_line.get("text") or ""),
                    "source_line_ids": [lid],
                    "amount_line_id": lid,
                    "label_line_id": lid,
                    "y_center": amount_line.get("y_center"),
                    "hint_tags": _tags_for_text(str(amount_line.get("text") or ""), a.get("value")),
                }
            )
            used_amount_ids.add(lid)
            used_label_ids.add(lid)
            continue

        ax, ay = _line_center(amount_line)
        candidates: list[dict[str, Any]] = []
        for label in label_lines:
            lab_id = str(label.get("line_id") or "")
            if lab_id == lid:
                continue
            lx, ly = _line_center(label)
            if abs(ly - ay) <= row_threshold * 1.35 and lx < ax:
                candidates.append(label)
        if not candidates:
            # Also allow a slightly wider below/above match, because OCR sometimes orders
            # right-column amount before the left-column label on the same printed row.
            for label in label_lines:
                lab_id = str(label.get("line_id") or "")
                if lab_id == lid:
                    continue
                lx, ly = _line_center(label)
                if abs(ly - ay) <= row_threshold * 2.1 and lx < ax:
                    candidates.append(label)
        if candidates:
            label = sorted(candidates, key=lambda c: _label_score(c, amount_line, row_threshold))[0]
            lab_id = str(label.get("line_id") or "")
            a = (amount_line.get("amount_candidates") or [])[-1]
            combined_text = f"{label.get('text')} | {amount_line.get('text')}"
            tags = sorted(
                set(
                    _tags_for_text(
                        str(label.get("text") or "") + " " + str(amount_line.get("text") or ""),
                        a.get("value"),
                    )
                )
            )
            if not any(
                t in tags
                for t in [
                    "total_keyword",
                    "net_keyword",
                    "tax_keyword",
                    "percentage_candidate",
                    "payment_keyword",
                    "change_keyword",
                    "discount_keyword",
                    "price_override_keyword",
                    "quantity_unit_price_note",
                    "date_candidate",
                    "time_candidate",
                ]
            ):
                tags.append("item_candidate")
            rows.append(
                {
                    "row_id": f"layout_row_{len(rows):03d}",
                    "evidence_kind": "spatial_label_amount_pair",
                    "left_text": str(label.get("text") or ""),
                    "middle_text": "",
                    "right_amount_raw": amount_line.get("text"),
                    "right_amount_value": a.get("value"),
                    "full_text": combined_text,
                    "source_line_ids": [lab_id, lid],
                    "label_line_id": lab_id,
                    "amount_line_id": lid,
                    "y_center": round((ay + _line_center(label)[1]) / 2.0, 4),
                    "hint_tags": tags,
                }
            )
            used_label_ids.add(lab_id)
            used_amount_ids.add(lid)
        else:
            a = (amount_line.get("amount_candidates") or [])[-1]
            rows.append(
                {
                    "row_id": f"layout_row_{len(rows):03d}",
                    "evidence_kind": "unpaired_amount_line",
                    "left_text": "",
                    "middle_text": "",
                    "right_amount_raw": amount_line.get("text"),
                    "right_amount_value": a.get("value"),
                    "full_text": str(amount_line.get("text") or ""),
                    "source_line_ids": [lid],
                    "amount_line_id": lid,
                    "label_line_id": None,
                    "y_center": amount_line.get("y_center"),
                    "hint_tags": _tags_for_text(str(amount_line.get("text") or ""), a.get("value"))
                    + ["unpaired_amount"],
                }
            )
            used_amount_ids.add(lid)

    # Add label-only lines not already represented, to preserve context around product blocks.
    for line in enriched:
        lid = str(line.get("line_id") or "")
        if lid in used_label_ids or lid in used_amount_ids:
            continue
        text = str(line.get("text") or "")
        rows.append(
            {
                "row_id": f"layout_row_{len(rows):03d}",
                "evidence_kind": "label_or_context_only",
                "left_text": text,
                "middle_text": "",
                "right_amount_raw": None,
                "right_amount_value": None,
                "full_text": text,
                "source_line_ids": [lid],
                "label_line_id": lid,
                "amount_line_id": None,
                "y_center": line.get("y_center"),
                "hint_tags": _tags_for_text(text),
            }
        )

    rows.sort(key=lambda r: (_num(r.get("y_center")), str(r.get("row_id"))))
    # Re-id after sorting and attach before/after context.
    for i, row in enumerate(rows):
        row["row_id"] = f"layout_row_{i:03d}"
    for i, row in enumerate(rows):
        if i > 0:
            row["above_row"] = {
                "row_id": rows[i - 1]["row_id"],
                "text": rows[i - 1].get("full_text"),
                "amount": rows[i - 1].get("right_amount_value"),
            }
        else:
            row["above_row"] = None
        if i + 1 < len(rows):
            row["below_row"] = {
                "row_id": rows[i + 1]["row_id"],
                "text": rows[i + 1].get("full_text"),
                "amount": rows[i + 1].get("right_amount_value"),
            }
        else:
            row["below_row"] = None

    candidate_rows = []
    for row in rows:
        tags = set(row.get("hint_tags") or [])
        if row.get("right_amount_value") is None:
            continue
        kind = None
        # Priority matters: discount/change/payment semantics must win over a
        # generic word like "Total". Example: "IKEA FAMILY Rabatt Total" is a
        # discount candidate, not the receipt grand total.
        if "change_keyword" in tags:
            kind = "change_candidate"
        elif "payment_keyword" in tags:
            kind = "payment_candidate"
        elif "discount_keyword" in tags or "negative_amount" in tags:
            kind = "discount_candidate"
        elif "total_keyword" in tags and "net_keyword" in tags:
            kind = "net_total_candidate_not_grand_total"
        elif "tax_keyword" in tags:
            kind = "tax_candidate"
        elif "total_keyword" in tags:
            kind = "grand_total_candidate"
        elif "price_override_keyword" in tags:
            kind = "discounted_item_price_candidate"
        elif "quantity_unit_price_note" in tags:
            kind = "quantity_unit_price_note"
        elif "item_candidate" in tags:
            kind = "item_price_candidate"
        if kind:
            candidate_rows.append(
                {
                    "kind": kind,
                    "row_id": row.get("row_id"),
                    "text": row.get("full_text"),
                    "amount": row.get("right_amount_value"),
                    "source_line_ids": row.get("source_line_ids"),
                    "hint_tags": row.get("hint_tags"),
                }
            )

    amount_neighbors = []
    id_to_index = {row.get("row_id"): i for i, row in enumerate(rows)}
    for row in rows:
        if row.get("right_amount_value") is None:
            continue
        i = id_to_index.get(row.get("row_id"), 0)
        amount_neighbors.append(
            {
                "row_id": row.get("row_id"),
                "amount": row.get("right_amount_value"),
                "text": row.get("full_text"),
                "source_line_ids": row.get("source_line_ids"),
                "previous_text": rows[i - 1].get("full_text") if i > 0 else None,
                "next_text": rows[i + 1].get("full_text") if i + 1 < len(rows) else None,
                "hint_tags": row.get("hint_tags"),
            }
        )

    def fmt_amount(v: Any) -> str:
        return "" if v is None else f"{float(v):.2f}"

    layout_view_lines = []
    for row in rows:
        tags = ",".join(row.get("hint_tags") or [])
        left = str(row.get("left_text") or "")[:54]
        right = str(row.get("right_amount_raw") or "")[:24]
        amount = fmt_amount(row.get("right_amount_value"))
        layout_view_lines.append(
            f"[{row['row_id']}] {left:54s} | {right:24s} | value={amount:>8s} | lines={','.join(row.get('source_line_ids') or [])} | {tags}"
        )

    return {
        "schema_version": "v14_1_layout_context_1",
        "row_threshold_norm": round(row_threshold, 5),
        "aligned_rows": rows,
        "layout_view": "\n".join(layout_view_lines),
        "amount_neighbors": amount_neighbors,
        "semantic_candidate_hints": candidate_rows,
    }
