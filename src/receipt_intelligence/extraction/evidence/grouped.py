#!/usr/bin/env python3
"""
Generic grouped-evidence builder.

This module creates receipt-pattern evidence for the LLM without deciding final
receipt data. It is intentionally merchant-agnostic: it detects common layout
patterns such as original-vs-final prices, quantity/unit/line-total groups,
payment/change relationships, and tax-table-like structures.

The output is prompt evidence only. The LLM still performs semantic extraction;
the deterministic validator still decides whether the LLM result can be imported.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

AMOUNT_TOL = 0.03

FINAL_PRICE_RE = re.compile(
    r"\b(IHR\s+PREIS|DEIN\s+PREIS|AKTIONSPREIS|ENDPREIS|SALE\s*PRICE|FINAL\s*PRICE|YOUR\s*PRICE|NOW\s*PRICE|REDUZIERT|REDUCED|SONDERPREIS)\b",
    re.IGNORECASE,
)
ORIGINAL_PRICE_RE = re.compile(
    r"\b(UVP|ORIGINAL|ORIGINALPREIS|LISTENPREIS|ALTER\s+PREIS|OLD\s*PRICE|WAS|STATT|NORMALPREIS)\b",
    re.IGNORECASE,
)
DISCOUNT_RE = re.compile(
    r"\b(RABATT|DISCOUNT|COUPON|AKTION|NACHLASS|GUTSCHRIFT|VOUCHER|BONUS|PROMO|PROMOTION)\b",
    re.IGNORECASE,
)
TOTAL_RE = re.compile(
    r"\b(SUMME|BONSUMME|BON\s*SUMME|GESAMT|TOTAL|ENDSUMME|ZU\s*ZAHLEN|BETRAG\s*FÄLLIG|AMOUNT\s*DUE|BALANCE\s*DUE)\b",
    re.IGNORECASE,
)
NET_RE = re.compile(r"\b(NETTO|NET|OHNE\s+MWST|OHNE\s+UST|TAXABLE)\b", re.IGNORECASE)
TAX_RE = re.compile(r"\b(MWST|M\.?W\.?ST|UST|U\.?ST|STEUER|VAT|TAX|IVA|TVA)\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"\b\d{1,2}(?:[,.]\d)?\s*%")
GROSS_RE = re.compile(r"\b(BRUTTO|GROSS|INKL\.?|INCL\.?|MIT\s+MWST)\b", re.IGNORECASE)
NON_AMBIGUOUS_GROSS_RE = re.compile(r"\b(BRUTTO|INKL\.?|INCL\.?|MIT\s+MWST)\b", re.IGNORECASE)
PAYMENT_RE = re.compile(
    r"\b(BAR|CASH|GEGEBEN|PAID|ZAHLUNG|KARTENZAHLUNG|LASTSCHRIFT|EC|GIROCARD|KARTE|CARD|VISA|MASTERCARD|MAESTRO|DEBIT|CREDIT|PAYPAL)\b",
    re.IGNORECASE,
)
CHANGE_RE = re.compile(
    r"\b(R[ÜUO]CKGELD|RUECKGELD|ROCKGELD|WECHSELGELD|CHANGE|CHANGE\s*DUE|RETURN\s*CASH)\b",
    re.IGNORECASE,
)
ARTICLE_CODE_RE = re.compile(r"^(?:ARTIKEL\s*)?\d{3,}$|^ARTIKEL\s+\d+", re.IGNORECASE)
NOISE_RE = re.compile(r"^(?:0|1|\*|EUR|EURO|€|[A-Z]?)$", re.IGNORECASE)
QUANTITY_LEFT_RE = re.compile(r"^\s*(\d{1,4})(?:[,.](\d{3}))?\s*$")
QUANTITY_NOTE_RE = re.compile(
    r"\b(\d+(?:[,.]\d+)?)\s*(?:STK|STÜCK|STUECK|PCS?|QTY|ANZ|KG|G|GRAMM|L|ML|PACK|PK)?\s*[x×*]\s*(\d{1,5}(?:[,.]\d{2}|\.\d{2}))\b",
    re.IGNORECASE,
)
RATE_RE = re.compile(r"(?<!\d)(\d{1,2}(?:[,.]\d+)?)\s*%")


def _txt(v: Any) -> str:
    return str(v or "").strip()


def _num(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return round(float(v), 2)
    except Exception:
        return None


def _rid(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or "")


def _sources(*rows: dict[str, Any] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for sid in row.get("source_line_ids") or []:
            s = str(sid)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _row_text(row: dict[str, Any]) -> str:
    left = _txt(row.get("left_text"))
    raw = _txt(row.get("right_amount_raw"))
    full = _txt(row.get("full_text"))
    if left and raw:
        return f"{left} | {raw}"
    return full or left or raw


def _is_context_name(row: dict[str, Any]) -> bool:
    text = _txt(row.get("left_text") or row.get("full_text"))
    if not text:
        return False
    if row.get("right_amount_value") is not None:
        return False
    if ARTICLE_CODE_RE.fullmatch(text) or NOISE_RE.fullmatch(text):
        return False
    if (
        PAYMENT_RE.search(text)
        or CHANGE_RE.search(text)
        or TAX_RE.search(text)
        or TOTAL_RE.search(text)
    ):
        return False
    if re.search(
        r"\b(?:DATUM|UHRZEIT|BELEG|TERMINAL|KUNDENBELEG|STEUER\s*-?NR|TEL|TELEFON|FILIALE)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    if len(text) < 4:
        return False
    return True


def _nearest_product_name(
    rows: list[dict[str, Any]], idx: int, window: int = 8
) -> dict[str, Any] | None:
    # Prefer nearest previous product-like text, then nearest next text. Skip article codes/noise.
    for direction in (-1, 1):
        for step in range(1, window + 1):
            j = idx + direction * step
            if not (0 <= j < len(rows)):
                continue
            row = rows[j]
            if _is_context_name(row):
                return row
    return None


def _parse_quantity_left(text: str) -> float | None:
    m = QUANTITY_LEFT_RE.fullmatch(_txt(text))
    if not m:
        return None
    try:
        # 12,000 on receipts often means quantity 12.000, not 12000.
        return float(f"{m.group(1)}.{m.group(2) or '0'}")
    except Exception:
        return None


def _parse_quantity_note(text: str) -> tuple[float, float, float] | None:
    m = QUANTITY_NOTE_RE.search(text or "")
    if not m:
        return None
    try:
        qty = float(m.group(1).replace(",", "."))
        unit = float(m.group(2).replace(",", "."))
        return qty, unit, round(qty * unit, 2)
    except Exception:
        return None


def _parse_quantity_note_row(row: dict[str, Any]) -> tuple[float, float, float] | None:
    # Handles rows split as left="2 ×" and right_amount_value=0.89 as well
    # as single text rows like "2 x 0,89".
    text = _row_text(row)
    parsed = _parse_quantity_note(text)
    if parsed:
        return parsed
    left = _txt(row.get("left_text"))
    m = re.search(
        r"\b(\d+(?:[,.]\d+)?)\s*(?:STK|STÜCK|STUECK|PCS?|QTY|ANZ|KG|G|GRAMM|L|ML|PACK|PK)?\s*[x×*]\s*$",
        left,
        re.IGNORECASE,
    )
    unit = _num(row.get("right_amount_value"))
    if m and unit is not None:
        try:
            qty = float(m.group(1).replace(",", "."))
            return qty, unit, round(qty * unit, 2)
        except Exception:
            return None
    return None


def _format_amount(v: Any) -> str:
    n = _num(v)
    return "null" if n is None else f"{n:.2f}"


def build_original_final_price_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        value = _num(row.get("right_amount_value"))
        left = _txt(row.get("left_text"))
        if value is None or not left:
            continue
        if FINAL_PRICE_RE.search(left):
            continue
        if (
            TOTAL_RE.search(left)
            or TAX_RE.search(left)
            or PAYMENT_RE.search(left)
            or CHANGE_RE.search(left)
            or DISCOUNT_RE.search(left)
        ):
            continue
        # Find nearby final-price indicator after the candidate original/list row.
        for j in range(i + 1, min(len(rows), i + 6)):
            nxt = rows[j]
            final_value = _num(nxt.get("right_amount_value"))
            if final_value is None:
                continue
            ntext = _row_text(nxt)
            if FINAL_PRICE_RE.search(ntext):
                candidates.append(
                    {
                        "candidate_id": f"orig_final_{len(candidates):03d}",
                        "pattern": "nearby_original_price_and_final_sale_price",
                        "description_candidate": left,
                        "original_or_reference_price": value,
                        "final_sale_price_candidate": final_value,
                        "price_delta": round(final_value - value, 2),
                        "row_ids": [_rid(row), _rid(nxt)],
                        "source_line_ids": _sources(row, nxt),
                        "evidence_text": f"{_row_text(row)} -> {_row_text(nxt)}",
                        "generic_rule": "Use the final-sale/your-price amount for the item; do not output original/list and final prices as separate items.",
                    }
                )
                break
    return candidates


def build_quantity_note_link_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        text = _row_text(row)
        parsed = _parse_quantity_note_row(row)
        if not parsed:
            continue
        qty, unit_price, computed_total = parsed
        matches = []
        for j in range(max(0, i - 3), min(len(rows), i + 5)):
            if j == i:
                continue
            r = rows[j]
            val = _num(r.get("right_amount_value"))
            if val is None:
                continue
            if abs(val - computed_total) <= AMOUNT_TOL:
                matches.append(
                    {
                        "row_id": _rid(r),
                        "text": _row_text(r),
                        "amount": val,
                        "source_line_ids": r.get("source_line_ids") or [],
                    }
                )
        candidates.append(
            {
                "candidate_id": f"qty_note_{len(candidates):03d}",
                "pattern": "quantity_unit_price_note_explains_adjacent_line_total",
                "quantity": qty,
                "unit_price": unit_price,
                "computed_total": computed_total,
                "note_row_id": _rid(row),
                "note_text": text,
                "nearby_matching_line_total_rows": matches,
                "source_line_ids": _sources(row),
                "generic_rule": "Do not output this note as a separate item when it explains a nearby product line total.",
            }
        )
    return candidates


def build_quantity_price_group_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        qty = _parse_quantity_left(_txt(row.get("left_text")))
        value = _num(row.get("right_amount_value"))
        if qty is None or value is None:
            i += 1
            continue
        # Gather consecutive/small-window rows with same quantity on left.
        group = [row]
        j = i + 1
        while j < len(rows) and j <= i + 4:
            qj = _parse_quantity_left(_txt(rows[j].get("left_text")))
            vj = _num(rows[j].get("right_amount_value"))
            if qj is not None and abs(qj - qty) <= 0.0001 and vj is not None:
                group.append(rows[j])
            elif _txt(rows[j].get("left_text")) in {"*", "0", "1"} or _txt(
                rows[j].get("full_text")
            ) in {"*", "0", "1"}:
                pass
            elif group:
                # tolerate one non-value separator but stop after substantial text
                if _is_context_name(rows[j]):
                    break
            j += 1
        if len(group) >= 2:
            values = sorted(
                [
                    _num(g.get("right_amount_value"))
                    for g in group
                    if _num(g.get("right_amount_value")) is not None
                ]
            )
            unit_price = None
            line_total = None
            for v in values:
                if v is None:
                    continue
                if abs(qty * v - max(values)) <= max(AMOUNT_TOL, 0.05):
                    unit_price = v
                    line_total = round(qty * v, 2)
                    break
            if line_total is None:
                # Fallback evidence only: smallest is likely unit, largest likely extended total.
                unit_price = values[0]
                line_total = values[-1]
            name_row = _nearest_product_name(rows, i, window=8)
            candidates.append(
                {
                    "candidate_id": f"qty_group_{len(candidates):03d}",
                    "pattern": "nearby_quantity_unit_price_line_total_group",
                    "description_candidate": _txt(name_row.get("left_text") if name_row else None)
                    or None,
                    "quantity": qty,
                    "unit_price_candidate": unit_price,
                    "line_total_candidate": line_total,
                    "row_ids": [_rid(g) for g in group] + ([_rid(name_row)] if name_row else []),
                    "source_line_ids": _sources(*(group + ([name_row] if name_row else []))),
                    "evidence_text": " ; ".join(
                        _row_text(g) for g in ([name_row] if name_row else []) + group
                    ),
                    "generic_rule": "Treat quantity, unit price and extended total as one item candidate; do not import unit price and line total as separate items.",
                }
            )
            i = max(j, i + 1)
        else:
            i += 1
    return candidates


def build_total_payment_change_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = []
    payments = []
    changes = []
    for row in rows:
        text = _row_text(row)
        val = _num(row.get("right_amount_value"))
        if val is None:
            continue
        if TOTAL_RE.search(text) and not NET_RE.search(text) and not DISCOUNT_RE.search(text):
            totals.append(row)
        if PAYMENT_RE.search(text):
            payments.append(row)
        if CHANGE_RE.search(text):
            changes.append(row)
    out: list[dict[str, Any]] = []
    for t in totals:
        gt = _num(t.get("right_amount_value"))
        for p in payments:
            pv = _num(p.get("right_amount_value"))
            if gt is None or pv is None:
                continue
            for c in changes or [None]:
                cv = _num(c.get("right_amount_value")) if c else None
                relationship = None
                if cv is not None and abs((pv - abs(cv)) - gt) <= max(AMOUNT_TOL, 0.05):
                    relationship = "payment_amount_minus_abs_change_equals_grand_total"
                elif abs(pv - gt) <= max(AMOUNT_TOL, 0.05):
                    relationship = "payment_amount_equals_grand_total"
                if relationship:
                    out.append(
                        {
                            "candidate_id": f"total_payment_{len(out):03d}",
                            "pattern": relationship,
                            "grand_total_candidate": gt,
                            "payment_amount_candidate": pv,
                            "change_candidate": cv,
                            "row_ids": [_rid(x) for x in [t, p, c] if x],
                            "source_line_ids": _sources(t, p, c),
                            "evidence_text": " ; ".join(_row_text(x) for x in [t, p, c] if x),
                            "generic_rule": "Grand total is amount due; payment may equal total, or cash payment minus change may equal total.",
                        }
                    )
    # Generic nearby cash-paid inference: sometimes the amount is printed above
    # a change row without a clear payment keyword on the same OCR line.
    for t in totals:
        gt = _num(t.get("right_amount_value"))
        if gt is None:
            continue
        try:
            ti = rows.index(t)
        except ValueError:
            ti = -1
        for c in changes:
            cv = _num(c.get("right_amount_value"))
            if cv is None:
                continue
            try:
                ci = rows.index(c)
            except ValueError:
                ci = -1
            lo = max(0, min(ti if ti >= 0 else ci, ci if ci >= 0 else ti) - 8)
            hi = min(len(rows), max(ti if ti >= 0 else ci, ci if ci >= 0 else ti) + 3)
            for p in rows[lo:hi]:
                if p is t or p is c:
                    continue
                pv = _num(p.get("right_amount_value"))
                if pv is None:
                    continue
                if abs((pv - abs(cv)) - gt) <= max(AMOUNT_TOL, 0.05):
                    out.append(
                        {
                            "candidate_id": f"total_payment_{len(out):03d}",
                            "pattern": "nearby_amount_minus_abs_change_equals_grand_total",
                            "grand_total_candidate": gt,
                            "payment_amount_candidate": pv,
                            "change_candidate": abs(cv),
                            "row_ids": [_rid(x) for x in [t, p, c] if x],
                            "source_line_ids": _sources(t, p, c),
                            "evidence_text": " ; ".join(_row_text(x) for x in [t, p, c] if x),
                            "generic_rule": "When change is printed, a nearby amount satisfying nearby_amount - abs(change) = grand_total is a cash-paid candidate, even if the payment label is on another line.",
                        }
                    )
    return out


def _header_lines_for_tax(
    rows: list[dict[str, Any]], start: int, window: int = 5
) -> list[dict[str, Any]]:
    out = []
    for j in range(max(0, start - window), start):
        text = _row_text(rows[j])
        if TAX_RE.search(text) or NET_RE.search(text) or GROSS_RE.search(text):
            out.append(rows[j])
    return out


def _rate_from_text(text: str) -> float | None:
    m = RATE_RE.search(text or "")
    if not m:
        return None
    try:
        return round(float(m.group(1).replace(",", ".")), 2)
    except Exception:
        return None


def build_tax_table_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Group rows by nearby rate text, then provide all values as evidence. We do
    # not force gross/net/tax mapping unless headers make it obvious.
    by_rate: dict[float, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for i, row in enumerate(rows):
        text = _row_text(row)
        val = _num(row.get("right_amount_value"))
        if val is None:
            continue
        # A percentage in product text (e.g. "MOZZARELLA 40%") is not a
        # tax table. Require explicit tax/VAT/MwSt/USt/net/gross context in the
        # row or nearby header before building tax evidence.
        if not TAX_RE.search(text):
            headers = _header_lines_for_tax(rows, i, window=6)
            header_text_probe = " ; ".join(_row_text(h) for h in headers)
            if not TAX_RE.search(header_text_probe):
                continue
        if DISCOUNT_RE.search(text):
            continue
        rate = _rate_from_text(text)
        if rate is None:
            # Look for nearby rate row sharing same table area.
            for j in range(max(0, i - 3), min(len(rows), i + 4)):
                rate = _rate_from_text(_row_text(rows[j]))
                if rate is not None:
                    break
        if rate is not None:
            # Do not treat a rate marker itself (e.g. OCR split "7,00 %") as a tax amount.
            # It remains available as neutral row evidence, but it should not
            # become tax_amount_candidate evidence.
            if (
                val is not None
                and abs(val - rate) <= 0.01
                and not any(k in text.upper() for k in ["MWST", "UST", "VAT", "TAX", "STEUER"])
            ):
                continue
            by_rate[rate].append((i, row))
    candidates: list[dict[str, Any]] = []
    for rate, pairs in sorted(by_rate.items()):
        if not pairs:
            continue
        pairs = sorted(pairs, key=lambda x: x[0])
        vals = [_num(r.get("right_amount_value")) for _, r in pairs]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        first_i = pairs[0][0]
        headers = _header_lines_for_tax(rows, first_i, window=6)
        header_text = " ; ".join(_row_text(h) for h in headers)
        role_hint = "unknown_tax_table_columns"
        gross = net = tax = None
        header_upper = (header_text or "").upper()
        # Generic inference when three values exist and one is consistent as tax = gross - net.
        if len(vals) >= 3:
            for a in vals:
                for b in vals:
                    for c in vals:
                        if a == b or a == c or b == c:
                            continue
                        if (
                            abs((a - b) - c) <= max(AMOUNT_TOL, 0.05)
                            and a > b
                            and c >= 0
                            and c <= min(a, b)
                        ):
                            gross, net, tax = a, b, c
                            role_hint = "gross_minus_net_matches_tax"
                            break
                    if role_hint != "unknown_tax_table_columns":
                        break
                if role_hint != "unknown_tax_table_columns":
                    break
        elif (
            len(vals) == 2
            and ("NETTO" in header_upper or "NET" in header_upper)
            and (
                "STEUER" in header_upper
                or "MWST" in header_upper
                or "VAT" in header_upper
                or "TAX" in header_upper
            )
        ):
            # Common compact tax summary: one net amount and one tax amount; gross is the grand total elsewhere.
            tax = min(vals)
            net = max(vals)
            role_hint = "net_and_tax_columns_detected"
        elif len(vals) == 1:
            tax = vals[0]
            role_hint = "single_tax_amount_row"
        candidates.append(
            {
                "candidate_id": f"tax_table_{len(candidates):03d}",
                "pattern": "tax_table_or_tax_summary_row",
                "rate_candidate": rate,
                "values_found": vals,
                "gross_candidate": gross,
                "net_candidate": net,
                "tax_amount_candidate": tax,
                "role_hint": role_hint,
                "header_context": header_text or None,
                "row_ids": [_rid(r) for _, r in pairs] + [_rid(h) for h in headers],
                "source_line_ids": _sources(*([r for _, r in pairs] + headers)),
                "evidence_text": " ; ".join(_row_text(r) for _, r in pairs),
                "generic_rule": "In tax tables, tax_amount_candidate is the tax/MwSt/VAT amount. tax_total should equal the sum of tax_amount_candidate values. Never use the tax rate, gross, or net value as tax_total. One tax table candidate normally becomes one tax object.",
            }
        )
    return candidates


def build_discount_application_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect generic cases where a visible discount may already be reflected.

    This does not decide whether to include/exclude the discount. It tells the
    LLM to reconcile both possibilities: final-sale prices may already include
    the general discount, while original/reference prices may require it.
    """
    final_price_rows: list[dict[str, Any]] = []
    discount_rows: list[dict[str, Any]] = []
    for row in rows:
        val = _num(row.get("right_amount_value"))
        text = _row_text(row)
        if val is None:
            continue
        if FINAL_PRICE_RE.search(text):
            final_price_rows.append(row)
        if DISCOUNT_RE.search(text) or val < 0:
            discount_rows.append(row)
    candidates: list[dict[str, Any]] = []
    for drow in discount_rows:
        dval = _num(drow.get("right_amount_value"))
        if dval is None:
            continue
        nearby_final_rows = []
        try:
            d_index = rows.index(drow)
        except ValueError:
            d_index = -1
        for frow in final_price_rows:
            try:
                f_index = rows.index(frow)
            except ValueError:
                f_index = 10**9
            if d_index < 0 or abs(f_index - d_index) <= 12:
                nearby_final_rows.append(frow)
        candidates.append(
            {
                "candidate_id": f"discount_application_{len(candidates):03d}",
                "pattern": "general_discount_near_final_sale_prices",
                "discount_amount_candidate": dval,
                "nearby_final_price_row_count": len(nearby_final_rows),
                "row_ids": [_rid(drow)] + [_rid(r) for r in nearby_final_rows[:8]],
                "source_line_ids": _sources(drow, *nearby_final_rows[:8]),
                "evidence_text": " ; ".join(
                    [_row_text(drow)] + [_row_text(r) for r in nearby_final_rows[:8]]
                ),
                "generic_rule": "If final-sale prices are used and those item totals already reconcile to the grand total, do not subtract this general discount again; treat it as already applied/informational. If original/list prices are used, the discount may be needed to reconcile.",
            }
        )
    return candidates


def build_semantic_row_context_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose ambiguous rows to the semantic LLM without classifying them.

    The deterministic evidence layer records geometry, amount presence and weak
    lexical hints only. It must not decide that a row is an item, footer, tax,
    payment or note row.
    """
    candidates: list[dict[str, Any]] = []
    for row in rows:
        text = _row_text(row)
        if not text:
            continue
        candidates.append(
            {
                "candidate_id": f"semantic_row_{len(candidates):03d}",
                "pattern": "row_requires_receipt_wide_semantic_classification",
                "amount_candidate": _num(row.get("right_amount_value")),
                "row_ids": [_rid(row)],
                "source_line_ids": _sources(row),
                "evidence_text": text,
                "hint_tags": sorted(set(row.get("hint_tags") or [])),
                "generic_rule": (
                    "Classify this row using the complete receipt structure. Keywords such as "
                    "Summe, Total, MwSt, Menge or Rabatt are not authoritative in isolation; "
                    "they may occur in headers, products, notes, totals or tax sections."
                ),
            }
        )
    return candidates


def build_amount_only_product_attachment_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach weak amount rows like '1 | 0,95' to nearby product-name rows.

    Evidence only. This helps receipts where OCR split a quantity marker, product
    name, and amount into separate rows.
    """
    candidates: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        val = _num(row.get("right_amount_value"))
        if val is None:
            continue
        left = _txt(row.get("left_text"))
        text = _row_text(row)
        if (
            PAYMENT_RE.search(text)
            or CHANGE_RE.search(text)
            or TAX_RE.search(text)
            or TOTAL_RE.search(text)
            or DISCOUNT_RE.search(text)
        ):
            continue
        weak_left = (not left) or re.fullmatch(r"\s*(?:\d{1,3}|[*x×])\s*", left) is not None
        if not weak_left:
            continue
        name_row = _nearest_product_name(rows, i, window=5)
        if not name_row:
            continue
        candidates.append(
            {
                "candidate_id": f"amount_attach_{len(candidates):03d}",
                "pattern": "amount_or_quantity_marker_row_attached_to_nearby_product_name",
                "description_candidate": _txt(
                    name_row.get("left_text") or name_row.get("full_text")
                ),
                "line_total_candidate": val,
                "row_ids": [_rid(name_row), _rid(row)],
                "source_line_ids": _sources(name_row, row),
                "evidence_text": f"{_row_text(name_row)} -> {text}",
                "generic_rule": "If an amount row has only a quantity/noise marker as left text, attach the amount to the nearest product-name row instead of ignoring it or treating the marker as the item name.",
            }
        )
    return candidates


def build_grouped_evidence(layout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in layout_rows or [] if isinstance(r, dict)]
    return {
        "schema_version": "v14_6_grouped_evidence_1",
        "original_final_price_candidates": build_original_final_price_candidates(rows),
        "quantity_note_link_candidates": build_quantity_note_link_candidates(rows),
        "quantity_price_group_candidates": build_quantity_price_group_candidates(rows),
        "total_payment_change_candidates": build_total_payment_change_candidates(rows),
        "tax_table_candidates": build_tax_table_candidates(rows),
        "discount_application_candidates": build_discount_application_candidates(rows),
        "semantic_row_context_candidates": build_semantic_row_context_candidates(rows),
        "amount_only_product_attachment_candidates": build_amount_only_product_attachment_candidates(
            rows
        ),
    }


def grouped_evidence_to_prompt_text(grouped: dict[str, Any], *, max_per_group: int = 40) -> str:
    if not grouped:
        return ""
    sections: list[str] = []
    labels = [
        (
            "original_final_price_candidates",
            "GENERIC ORIGINAL/LIST PRICE VS FINAL-PRICE CANDIDATES",
        ),
        ("quantity_note_link_candidates", "GENERIC QUANTITY-NOTE LINK CANDIDATES"),
        (
            "quantity_price_group_candidates",
            "GENERIC QUANTITY × UNIT PRICE × LINE TOTAL CANDIDATES",
        ),
        ("total_payment_change_candidates", "GENERIC TOTAL / PAYMENT / CHANGE RELATIONSHIPS"),
        ("tax_table_candidates", "GENERIC TAX-TABLE CANDIDATES"),
        ("discount_application_candidates", "GENERIC ALREADY-APPLIED DISCOUNT CANDIDATES"),
        ("semantic_row_context_candidates", "ROWS FOR RECEIPT-WIDE SEMANTIC CLASSIFICATION"),
        (
            "amount_only_product_attachment_candidates",
            "GENERIC AMOUNT-TO-PRODUCT ATTACHMENT CANDIDATES",
        ),
    ]
    for key, title in labels:
        items = grouped.get(key) or []
        if not items:
            continue
        lines = [title + ":"]
        for item in items[:max_per_group]:
            cid = item.get("candidate_id")
            pattern = item.get("pattern")
            evidence = _txt(item.get("evidence_text"))
            rule = _txt(item.get("generic_rule"))
            core_parts = []
            for field in [
                "description_candidate",
                "quantity",
                "unit_price_candidate",
                "line_total_candidate",
                "original_or_reference_price",
                "final_sale_price_candidate",
                "grand_total_candidate",
                "payment_amount_candidate",
                "change_candidate",
                "rate_candidate",
                "gross_candidate",
                "net_candidate",
                "tax_amount_candidate",
                "discount_amount_candidate",
                "nearby_final_price_row_count",
                "amount_candidate",
            ]:
                if item.get(field) is not None:
                    core_parts.append(f"{field}={item.get(field)}")
            lines.append(
                f"- [{cid}] {pattern}; {'; '.join(core_parts)}; rows={item.get('row_ids')}; lines={item.get('source_line_ids')}; evidence={evidence}; rule={rule}"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
