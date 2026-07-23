#!/usr/bin/env python3
"""
Deterministic receipt validation.

Architecture:
    LLM receipt JSON + OCR context -> validation report only

This module deliberately does NOT repair or reparse the receipt. It checks math,
structure, evidence coverage, OCR confidence, and produces a failure diagnosis.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from receipt_intelligence.extraction.evidence.grouped import build_grouped_evidence

# Validator amount parsing is intentionally stricter for OCR text than for
# normalized JSON numbers, so date fragments such as 09.12 are not treated as
# unresolved money. JSON numeric fields are passed as int/float and still work.
AMOUNT_RE = re.compile(
    r"(?<!\d)([-+−]?\s*\d{1,5}(?:[.\s]\d{3})*,\s*\d{2}|[-+−]?\s*\d{1,5}\.\d{2}|[-+−]?\s*\d{1,5}\s+\d{2})(?:\s*[-−])?(?!\d)"
)
DATE_RE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
NON_ITEM_HINT_RE = re.compile(
    r"\b(SUMME|ZWISCHENSUMME|SUBTOTAL|TOTAL|GESAMT|BETRAG|MWST|UST|NETTO|BRUTTO|STEUER|EC|GIROCARD|KARTE|CARD|VISA|MASTERCARD|BAR|CASH|GEGEBEN|ZAHLUNG|KARTENZAHLUNG|LASTSCHRIFT|RÜCKGELD|RUECKGELD|DATUM|UHRZEIT|BELEG|BON|TRACE|TERMINAL|KUNDENBELEG|KUNDENKARTE|PAYBACK|PUNKT|PUNKTE|PUNKTESTAND|BONUSPUNKTE|TREUEPUNKTE|LOYALTY|POINTS)\b",
    re.IGNORECASE,
)
CHANGE_LABEL_RE = re.compile(
    r"\b(RÜCKG|RUECKG|RÜCKGELD|RUECKGELD|WECHSELGELD|CHANGE|ZURÜCK|ZURUECK)\b", re.IGNORECASE
)
TENDER_LABEL_RE = re.compile(
    r"\b(BAR|BARGELD|CASH|GEGEBEN|ZAHLUNG|KARTE|CARD|GIROCARD|EC|VISA|MASTERCARD|COUPON|GUTSCHEIN|VOUCHER|RABATT[- ]?COUPON)\b",
    re.IGNORECASE,
)


DATE_TOKEN_RE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\b")
TIME_TOKEN_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
TAX_TABLE_GROSS_OR_TOTAL_CONTEXT_RE = re.compile(
    r"\b(?:TOTAL|GESAMT|BRUTTO|GROSS|INKL\.?\s*MWST|NETTO)\b", re.IGNORECASE
)


def normalize_date_token(value: Any) -> str | None:
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
            y = 2000 + y if y <= 49 else 1900 + y
    try:
        return date(y, mo, d).isoformat()
    except Exception:
        return None


def supported_date_candidates_from_context(ocr_context: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    containers = []
    containers.extend(ocr_context.get("date_time_candidates") or [])
    containers.extend(ocr_context.get("lines") or [])
    for row in containers:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("full_text") or "")
        for token in DATE_TOKEN_RE.findall(text):
            norm = normalize_date_token(token)
            if norm:
                out.add(norm)
    return out


def normalize_time_token(value: Any) -> str | None:
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


def supported_time_candidates_from_context(ocr_context: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    containers = []
    containers.extend(ocr_context.get("date_time_candidates") or [])
    containers.extend(ocr_context.get("lines") or [])
    for row in containers:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("full_text") or "")
        for m in TIME_TOKEN_RE.finditer(text):
            norm = normalize_time_token(m.group(0))
            if norm:
                out.add(norm)
    return out


def used_numeric_values_from_receipt(receipt: dict[str, Any]) -> set[float]:
    vals: set[float] = set()

    def add(v: Any) -> None:
        x = amount(v)
        if x is not None:
            vals.add(round(x, 2))

    totals = receipt.get("totals") if isinstance(receipt.get("totals"), dict) else {}
    for key in ("grand_total", "subtotal", "tax_total", "paid_total", "change"):
        add(totals.get(key))
    for item in receipt.get("items") or []:
        if isinstance(item, dict):
            add(item.get("line_total"))
            add(item.get("unit_price"))
    for payment in receipt.get("payments") or []:
        if isinstance(payment, dict):
            add(payment.get("amount"))
    for tax in receipt.get("taxes") or []:
        if isinstance(tax, dict):
            add(tax.get("tax"))
            add(tax.get("gross"))
            add(tax.get("net"))
    return vals


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def amount(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    raw = str(value).strip().replace("−", "-")
    if not raw:
        return None
    # Date/time-only OCR strings should not become monetary values.
    if (DATE_RE.search(raw) or TIME_RE.search(raw)) and "," not in raw:
        return None
    # Accept normalized LLM dot-decimal strings only when the whole string is a number.
    if re.fullmatch(r"[-+]?\d{1,7}\.\d{1,2}", raw):
        try:
            return round(float(raw), 2)
        except Exception:
            return None
    m = AMOUNT_RE.search(raw)
    if not m:
        return None
    token = m.group(0).strip()
    negative = (
        token.startswith("-")
        or token.endswith("-")
        or re.search(r",\s*\d{2}\s*[-−]", token) is not None
    )
    s = re.sub(r"[^0-9,\.\s]", "", token).strip().replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "." in s:
        pass
    else:
        return None
    try:
        v = round(float(s), 2)
    except Exception:
        return None
    return -abs(v) if negative else v


def _amount_close(a: float | None, b: float | None, tolerance: float = 0.05) -> bool:
    if a is None or b is None:
        return False
    return abs(round(abs(float(a)) - abs(float(b)), 2)) <= tolerance


def _line_amounts_from_container(row: dict[str, Any]) -> list[float]:
    values: list[float] = []
    candidate_lists = [
        row.get("amount_candidates"),
        row.get("amounts"),
        row.get("right_amounts"),
    ]
    for candidates in candidate_lists:
        if not isinstance(candidates, list):
            continue
        for cand in candidates:
            if isinstance(cand, dict):
                parsed = amount(
                    cand.get("value") if cand.get("value") is not None else cand.get("raw")
                )
            else:
                parsed = amount(cand)
            if parsed is not None:
                values.append(parsed)
    text = str(
        row.get("text")
        or row.get("row_text")
        or row.get("left_text")
        or row.get("right_text")
        or ""
    )
    for m in AMOUNT_RE.finditer(text):
        parsed = amount(m.group(0))
        if parsed is not None:
            values.append(parsed)
    # Deduplicate by rounded absolute value while preserving sign examples.
    out: list[float] = []
    seen: set[tuple[float, int]] = set()
    for value in values:
        key = (round(abs(value), 2), -1 if value < 0 else 1)
        if key not in seen:
            seen.add(key)
            out.append(round(value, 2))
    return out


def evidence_amounts_by_role(ocr_context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Collect printed tender/change amount evidence without interpreting the receipt.

    This is an evidence-support check only. It does not create or repair
    semantic values. It catches cases where the extracted change/payment value is
    a computed value that is not actually printed while a different printed
    change/tender amount exists.
    """
    roles = {"change": [], "tender": []}
    containers: list[dict[str, Any]] = []
    for key in ("lines", "layout_rows", "visual_evidence_lines"):
        for row in ocr_context.get(key) or []:
            if isinstance(row, dict):
                containers.append(row)
    for row in containers:
        text = str(
            row.get("text")
            or row.get("row_text")
            or row.get("left_text")
            or row.get("right_text")
            or ""
        )
        if not text:
            continue
        amounts = _line_amounts_from_container(row)
        if not amounts:
            continue
        lid = str(row.get("line_id") or row.get("row_id") or row.get("id") or "")
        record = {"line_id": lid, "text": text, "amounts": amounts}
        if CHANGE_LABEL_RE.search(text):
            roles["change"].append(record)
        if TENDER_LABEL_RE.search(text):
            roles["tender"].append(record)
    return roles


def source_line_text_for_ids(ocr_context: dict[str, Any], source_ids: list[Any]) -> str:
    wanted = {str(x) for x in (source_ids or []) if x is not None}
    texts: list[str] = []
    for row in (ocr_context.get("lines") or []) + (ocr_context.get("layout_rows") or []):
        if not isinstance(row, dict):
            continue
        ids = {str(row.get("line_id") or row.get("row_id") or row.get("id") or "")}
        ids.update(str(x) for x in (row.get("source_line_ids") or []) if x is not None)
        if wanted.intersection(ids):
            txt = str(
                row.get("text")
                or row.get("row_text")
                or row.get("full_text")
                or row.get("left_text")
                or ""
            ).strip()
            if txt:
                texts.append(txt)
    return " ; ".join(dict.fromkeys(texts))


def tax_source_looks_like_product_percent(tax: dict[str, Any], ocr_context: dict[str, Any]) -> bool:
    text = source_line_text_for_ids(ocr_context, tax.get("source_line_ids") or [])
    if not text:
        return False
    has_percent = bool(re.search(r"\b\d{1,2}(?:[,.]\d+)?\s*%", text))
    has_product_word = bool(re.search(r"[A-Za-zÄÖÜäöüß]{3,}", text))
    has_tax_context = bool(
        re.search(r"\b(MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|GROSS)\b", text, re.IGNORECASE)
    )
    return has_percent and has_product_word and not has_tax_context


def issue(code: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "details": details}


def _canonical_line_id(line_id: Any) -> str:
    sid = str(line_id or "").strip()
    m = re.fullmatch(r"(line|region_line)_(\d+)", sid)
    if m:
        return f"{m.group(1)}_{int(m.group(2)):03d}"
    return sid


def line_index(ocr_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a unified evidence-line index.

    V14.13+ LLM outputs may cite full OCR lines, layout rows, VLM table rows,
    and region crop re-OCR rows. Validation should check whether the cited
    evidence exists across all evidence registries, not only full-image OCR.
    """
    idx = {
        str(line.get("line_id")): line
        for line in (ocr_context.get("lines") or [])
        if isinstance(line, dict) and line.get("line_id")
    }
    for row in ocr_context.get("layout_rows") or []:
        if not isinstance(row, dict):
            continue
        rid = row.get("row_id") or row.get("id")
        if rid:
            idx[str(rid)] = {
                "line_id": str(rid),
                "text": row.get("row_text")
                or row.get("text")
                or row.get("left_text")
                or "layout row",
                "confidence": row.get("confidence", 1.0),
                "amount_candidates": row.get("amount_candidates") or [],
            }
    for line in ocr_context.get("visual_evidence_lines") or []:
        if not isinstance(line, dict):
            continue
        lid = line.get("line_id") or line.get("id") or line.get("row_id")
        if lid:
            idx[str(lid)] = {
                "line_id": str(lid),
                "text": line.get("text")
                or line.get("row_text")
                or line.get("description_candidate")
                or "visual evidence row",
                "confidence": line.get("confidence", line.get("layout_confidence", 1.0)),
                "amount_candidates": line.get("amount_candidates") or line.get("amounts") or [],
            }
    # Accept non-zero-padded OCR IDs such as line_03 when the registry uses line_003.
    for key, val in list(idx.items()):
        canon = _canonical_line_id(key)
        if canon and canon not in idx:
            idx[canon] = val
        m = re.fullmatch(r"(line|region_line)_0*(\d+)", key)
        if m:
            loose = f"{m.group(1)}_{int(m.group(2))}"
            idx.setdefault(loose, val)
    return idx


def collect_source_line_ids(receipt: dict[str, Any]) -> list[str]:
    ids: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, list):
            for x in value:
                if x is not None and str(x).strip():
                    ids.append(str(x))
        elif value is not None and str(value).strip():
            ids.append(str(value))

    merchant = receipt.get("merchant") if isinstance(receipt.get("merchant"), dict) else {}
    add(merchant.get("source_line_ids"))
    totals = receipt.get("totals") if isinstance(receipt.get("totals"), dict) else {}
    add(totals.get("source_line_ids"))
    for key in ("items", "taxes", "payments"):
        for row in receipt.get(key) or []:
            if isinstance(row, dict):
                add(row.get("source_line_ids"))
    return ids


def expand_used_line_ids_via_layout_rows(
    source_ids: list[str], ocr_context: dict[str, Any]
) -> set[str]:
    """Expand used source ids through V14 layout-row groups.

    If the LLM cites only a product-name line, the paired amount line from the
    same layout row should not be flagged as unresolved. This is validation-only
    evidence coverage, not semantic repair.
    """
    used = {str(x) for x in source_ids if str(x).strip()}
    changed = True
    rows = [r for r in (ocr_context.get("layout_rows") or []) if isinstance(r, dict)]
    while changed:
        changed = False
        for row in rows:
            group = {str(x) for x in (row.get("source_line_ids") or []) if str(x).strip()}
            if group and used.intersection(group) and not group.issubset(used):
                used.update(group)
                changed = True
    return used


def sum_item_totals(receipt: dict[str, Any]) -> tuple[float | None, int, int]:
    values: list[float] = []
    missing = 0
    for item in receipt.get("items") or []:
        if not isinstance(item, dict):
            continue
        v = amount(item.get("line_total"))
        if v is None:
            missing += 1
            continue
        values.append(v)
    if not values:
        return None, 0, missing
    return round(sum(values), 2), len(values), missing


def payment_sum(receipt: dict[str, Any]) -> tuple[float | None, int]:
    values = [amount(p.get("amount")) for p in receipt.get("payments") or [] if isinstance(p, dict)]
    values = [v for v in values if v is not None]
    if not values:
        return None, 0
    return round(sum(values), 2), len(values)


def tax_sum(receipt: dict[str, Any]) -> tuple[float | None, int]:
    values = [amount(t.get("tax")) for t in receipt.get("taxes") or [] if isinstance(t, dict)]
    values = [v for v in values if v is not None]
    if not values:
        return None, 0
    return round(sum(values), 2), len(values)


def positive_and_discount_item_sums(
    receipt: dict[str, Any],
) -> tuple[float | None, float | None, int]:
    positive: list[float] = []
    discounts: list[float] = []
    for item in receipt.get("items") or []:
        if not isinstance(item, dict):
            continue
        v = amount(item.get("line_total"))
        if v is None:
            continue
        category = str(item.get("category") or "").lower()
        if v < 0 or category == "discount":
            discounts.append(v)
        else:
            positive.append(v)
    pos = round(sum(positive), 2) if positive else None
    disc = round(sum(discounts), 2) if discounts else None
    return pos, disc, len(discounts)


def grouped_evidence(ocr_context: dict[str, Any]) -> dict[str, Any]:
    return build_grouped_evidence(
        [r for r in (ocr_context.get("layout_rows") or []) if isinstance(r, dict)]
    )


def explained_candidate_line_ids(ocr_context: dict[str, Any]) -> set[str]:
    """Lines already explained by generic grouped evidence.

    Examples: quantity notes, tax-table amount columns, and printed discounts
    that grouped evidence marks as potentially already applied should not become
    generic unresolved-amount warnings. This is evidence coverage only, not
    semantic parsing or correction.
    """
    out: set[str] = set()
    ge = grouped_evidence(ocr_context)
    for key in (
        "quantity_note_link_candidates",
        "tax_table_candidates",
        "discount_application_candidates",
    ):
        for cand in ge.get(key) or []:
            if not isinstance(cand, dict):
                continue
            for sid in cand.get("source_line_ids") or []:
                out.add(str(sid))
            for match in cand.get("nearby_matching_line_total_rows") or []:
                if isinstance(match, dict):
                    for sid in match.get("source_line_ids") or []:
                        out.add(str(sid))
    return out


def tax_table_tax_sum_from_evidence(ocr_context: dict[str, Any]) -> tuple[float | None, int]:
    ge = grouped_evidence(ocr_context)
    vals: list[float] = []
    common_rates = {0.0, 5.0, 7.0, 19.0, 20.0}
    for cand in ge.get("tax_table_candidates") or []:
        if not isinstance(cand, dict):
            continue
        v = amount(cand.get("tax_amount_candidate"))
        if v is None:
            # Some receipts print compact tax lines such as
            # "Mwst 07,00% = 1,43" where grouped evidence sees the rate and
            # tax amount in neighbouring rows but cannot assign table columns.
            # Use this fallback only when the context does not clearly identify
            # the lone non-rate value as gross/net/total. For fuel receipts like
            # "TOTAL inkl. MWSt" + "19.00 % MWSt | 59,62", 59,62 is gross,
            # not VAT, and must not be counted as tax-table tax evidence.
            found = []
            for raw in cand.get("values_found") or []:
                x = amount(raw)
                if x is not None:
                    found.append(x)
            non_rate_values = [x for x in found if round(abs(x), 2) not in common_rates]
            context = " ".join(
                str(cand.get(k) or "") for k in ("header_context", "evidence_text", "role_hint")
            )
            gross_or_total_context = bool(TAX_TABLE_GROSS_OR_TOTAL_CONTEXT_RE.search(context))
            if len(non_rate_values) == 1 and not gross_or_total_context:
                v = non_rate_values[0]
        if v is not None:
            vals.append(v)
    if not vals:
        return None, 0
    return round(sum(vals), 2), len(vals)


def likely_unresolved_amount_lines(
    receipt: dict[str, Any], ocr_context: dict[str, Any]
) -> list[dict[str, Any]]:
    used = expand_used_line_ids_via_layout_rows(collect_source_line_ids(receipt), ocr_context)
    unresolved = {
        str(x.get("line_id")) for x in receipt.get("unresolved_rows") or [] if isinstance(x, dict)
    }
    explained_by_grouped_candidates = explained_candidate_line_ids(ocr_context)
    used_values = used_numeric_values_from_receipt(receipt)
    suspicious: list[dict[str, Any]] = []
    for line in ocr_context.get("lines") or []:
        if not isinstance(line, dict):
            continue
        lid = str(line.get("line_id") or "")
        text = str(line.get("text") or "")
        if DATE_RE.search(text) or TIME_RE.search(text):
            continue
        amounts = []
        for cand in line.get("amount_candidates") or []:
            raw = str(cand.get("raw") or "")
            parsed = amount(raw)
            if parsed is not None:
                amounts.append({"raw": raw, "value": parsed, "span": cand.get("span") or []})
        if not lid or not amounts:
            continue
        if lid in used or lid in unresolved or lid in explained_by_grouped_candidates:
            continue
        if NON_ITEM_HINT_RE.search(text):
            continue
        # Region/VLM rows are valid evidence sources. When the LLM cites region_*
        # or VLM table rows, the corresponding full-image OCR amount line may not
        # share a source ID. If its numeric amount is already consumed by an
        # extracted field, do not report it as an unresolved amount line; arithmetic
        # mismatch checks still catch genuinely missing items.
        if used_values and any(
            any(abs(float(a["value"]) - v) <= max(0.03, 0.05) for v in used_values) for a in amounts
        ):
            continue
        # This remains only a diagnostic: it does not convert the row into an item.
        suspicious.append({"line_id": lid, "text": text, "amount_candidates": amounts})
    return suspicious


def diagnose_failure(
    parse_status: str, issues: list[dict[str, Any]], ocr_context: dict[str, Any]
) -> dict[str, Any]:
    codes = {i["code"] for i in issues}
    high = [i for i in issues if i.get("severity") in {"critical", "high"}]
    low_conf_count = len(ocr_context.get("low_confidence_lines") or [])
    line_count = int(ocr_context.get("line_count") or 0)
    low_conf_ratio = low_conf_count / max(line_count, 1)

    if parse_status == "failed":
        primary = "llm_main_parser_failed"
        explanation = (
            "The LLM did not return usable receipt JSON. No deterministic fallback was used."
        )
    elif "MISSING_TOTAL" in codes or "NO_ITEMS" in codes:
        if low_conf_ratio > 0.25:
            primary = "ocr_or_layout_quality"
            explanation = "Core receipt anchors are missing and many OCR lines have low confidence."
        else:
            primary = "llm_semantic_extraction"
            explanation = "OCR context exists, but the LLM omitted core receipt anchors."
    elif "ITEM_SUM_MISMATCH" in codes or "PAYMENT_TOTAL_MISMATCH" in codes:
        primary = "semantic_or_ocr_numeric_mismatch"
        explanation = "The extracted numbers do not reconcile with the printed total/payment. Check OCR numeric evidence and LLM row classification."
    elif high:
        primary = "validation_blocked_import"
        explanation = "High-severity validation issues prevent clean import."
    elif issues:
        primary = "needs_review"
        explanation = (
            "The receipt is structurally usable but has warnings that should be inspected."
        )
    else:
        primary = "ok"
        explanation = "No blocking validation issue was found."
    return {
        "primary_failure_mode": primary,
        "explanation": explanation,
        "low_confidence_line_count": low_conf_count,
        "line_count": line_count,
        "issue_codes": sorted(codes),
    }


def validate_receipt(
    receipt: dict[str, Any], ocr_context: dict[str, Any], tolerance: float = 0.03
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    parse_status = str(receipt.get("parse_status") or "partial")

    if parse_status == "failed":
        issues.append(
            issue(
                "LLM_PARSE_FAILED",
                "critical",
                "LLM main parser failed; deterministic fallback was intentionally not used.",
            )
        )

    merchant = receipt.get("merchant") if isinstance(receipt.get("merchant"), dict) else {}
    if not merchant.get("name"):
        issues.append(issue("MISSING_MERCHANT", "medium", "Merchant name is missing."))
    if not receipt.get("date"):
        issues.append(issue("MISSING_DATE", "medium", "Receipt date is missing."))
    else:
        extracted_date = normalize_date_token(receipt.get("date"))
        supported_dates = supported_date_candidates_from_context(ocr_context)
        if extracted_date is not None and supported_dates and extracted_date not in supported_dates:
            issues.append(
                issue(
                    "DATE_NOT_SUPPORTED_BY_OCR",
                    "medium",
                    "Extracted receipt date is not supported by OCR date evidence.",
                    extracted_date=extracted_date,
                    supported_dates=sorted(supported_dates),
                )
            )

    if receipt.get("time"):
        extracted_time = normalize_time_token(receipt.get("time"))
        supported_times = supported_time_candidates_from_context(ocr_context)
        if extracted_time is None:
            issues.append(
                issue(
                    "INVALID_TIME_VALUE",
                    "medium",
                    "Extracted receipt time has invalid hour/minute/second values.",
                    extracted_time=receipt.get("time"),
                    supported_times=sorted(supported_times),
                )
            )
        else:
            extracted_time_hhmm = extracted_time[:5]
            supported_hhmm = {t[:5] for t in supported_times}
            if (
                supported_times
                and extracted_time not in supported_times
                and extracted_time_hhmm not in supported_hhmm
            ):
                issues.append(
                    issue(
                        "TIME_NOT_SUPPORTED_BY_OCR",
                        "low",
                        "Extracted receipt time is not directly supported by OCR time evidence.",
                        extracted_time=extracted_time,
                        supported_times=sorted(supported_times),
                    )
                )

    totals = receipt.get("totals") if isinstance(receipt.get("totals"), dict) else {}
    grand_total = amount(totals.get("grand_total"))
    paid_total = amount(totals.get("paid_total"))
    change = amount(totals.get("change"))
    subtotal = amount(totals.get("subtotal"))
    tax_total = amount(totals.get("tax_total"))
    if grand_total is None:
        issues.append(issue("MISSING_TOTAL", "high", "Grand total is missing."))

    calculated_items, priced_item_count, missing_price_count = sum_item_totals(receipt)
    positive_items_total, discount_items_total, discount_item_count = (
        positive_and_discount_item_sums(receipt)
    )
    if priced_item_count == 0:
        issues.append(issue("NO_ITEMS", "high", "No priced item rows were extracted."))
    if missing_price_count:
        issues.append(
            issue(
                "ITEMS_WITHOUT_LINE_TOTAL",
                "medium",
                "Some extracted items have no line_total.",
                missing_count=missing_price_count,
            )
        )

    difference = None
    if calculated_items is not None and grand_total is not None:
        difference = round(calculated_items - grand_total, 2)
        if abs(difference) > tolerance:
            issues.append(
                issue(
                    "ITEM_SUM_MISMATCH",
                    "high",
                    "Sum of extracted item line totals does not match grand total.",
                    calculated_items=calculated_items,
                    grand_total=grand_total,
                    difference=difference,
                    tolerance=tolerance,
                )
            )

    if (
        calculated_items is not None
        and grand_total is not None
        and positive_items_total is not None
        and discount_items_total is not None
        and abs(calculated_items - grand_total) > tolerance
        and abs(positive_items_total - grand_total) <= max(tolerance, 0.05)
        and discount_item_count > 0
    ):
        issues.append(
            issue(
                "DISCOUNT_LIKELY_ALREADY_APPLIED",
                "high",
                "Extracted discount appears to have been subtracted twice: positive item sum already matches grand total.",
                positive_item_total=positive_items_total,
                discount_total=discount_items_total,
                calculated_items=calculated_items,
                grand_total=grand_total,
            )
        )

    pay_sum, payment_count = payment_sum(receipt)
    payment_amount_for_check = pay_sum if pay_sum is not None else paid_total
    payment_count_for_report = payment_count + (
        1 if pay_sum is None and paid_total is not None else 0
    )
    if change is not None and payment_amount_for_check is None:
        issues.append(
            issue(
                "CHANGE_WITHOUT_PAYMENT_AMOUNT",
                "medium",
                "Change was extracted but payment amount / paid_total is missing; cannot validate paid_total - change = grand_total.",
                change=change,
                grand_total=grand_total,
            )
        )
    if payment_amount_for_check is not None and grand_total is not None:
        payment_difference = round(payment_amount_for_check - grand_total, 2)
        cash_change_balances = False
        if change is not None:
            # Receipts may print change either as positive returned cash or as a
            # negative cash movement. Validate the semantic absolute change.
            cash_change_balances = abs(
                round(payment_amount_for_check - abs(change) - grand_total, 2)
            ) <= max(tolerance, 0.05)
        if abs(payment_difference) > max(tolerance, 0.05) and not cash_change_balances:
            issues.append(
                issue(
                    "PAYMENT_TOTAL_MISMATCH",
                    "medium",
                    "Payment amount does not match grand total. For cash receipts with change, payment_amount - abs(change) must equal grand_total.",
                    payment_sum=payment_amount_for_check,
                    grand_total=grand_total,
                    change=change,
                    difference=payment_difference,
                )
            )
        # Evidence-support check: if a printed change row exists, the extracted
        # semantic change should match that printed amount, not only a computed
        # payment-total difference. This catches missed coupon/voucher tenders.
        support = evidence_amounts_by_role(ocr_context)
        if change is not None and support.get("change"):
            printed_change_values = [v for row in support["change"] for v in row.get("amounts", [])]
            if printed_change_values and not any(
                _amount_close(change, v, max(tolerance, 0.05)) for v in printed_change_values
            ):
                issues.append(
                    issue(
                        "CHANGE_NOT_SUPPORTED_BY_PRINTED_CHANGE_LINE",
                        "medium",
                        "Extracted change does not match a printed change row. Check whether a coupon/voucher/tender was omitted from payments.",
                        extracted_change=change,
                        printed_change_values=printed_change_values[:10],
                        printed_change_lines=support["change"][:10],
                    )
                )
    elif payment_count_for_report == 0:
        sev = "medium" if change is not None else "low"
        issues.append(issue("NO_PAYMENT", sev, "No payment method/amount was extracted."))

    t_sum, tax_count = tax_sum(receipt)
    if tax_total is not None and t_sum is not None:
        tax_difference = round(t_sum - tax_total, 2)
        if abs(tax_difference) > max(tolerance, 0.05):
            issues.append(
                issue(
                    "TAX_SUM_MISMATCH",
                    "medium",
                    "Tax detail rows do not match extracted tax_total.",
                    tax_sum=t_sum,
                    tax_total=tax_total,
                    difference=tax_difference,
                )
            )
    table_tax_sum, table_tax_count = tax_table_tax_sum_from_evidence(ocr_context)
    if (
        table_tax_sum is not None
        and tax_total is not None
        and abs(table_tax_sum - tax_total) > max(tolerance, 0.05)
    ):
        issues.append(
            issue(
                "TAX_TOTAL_CONFLICTS_WITH_TAX_TABLE_EVIDENCE",
                "medium",
                "Extracted tax_total conflicts with structured tax-table evidence. tax_total should use tax/MwSt/VAT amounts, not rate/net/gross values.",
                tax_total=tax_total,
                tax_table_tax_sum=table_tax_sum,
                tax_table_count=table_tax_count,
            )
        )
    if table_tax_count and tax_count > max(table_tax_count + 1, table_tax_count * 2):
        issues.append(
            issue(
                "TAX_TABLE_OVER_SPLIT",
                "medium",
                "LLM appears to have split tax table columns into too many tax rows. One tax-table candidate normally maps to one tax object.",
                extracted_tax_count=tax_count,
                tax_table_candidate_count=table_tax_count,
            )
        )

    for tax in receipt.get("taxes") or []:
        if not isinstance(tax, dict):
            continue
        rate = amount(tax.get("rate"))
        tax_value = amount(tax.get("tax"))
        gross = amount(tax.get("gross"))
        if rate is not None and rate not in {0.0, 5.0, 7.0, 19.0} and not (0.0 <= rate <= 25.0):
            if tax_source_looks_like_product_percent(tax, ocr_context):
                issues.append(
                    issue(
                        "TAX_RATE_FROM_PRODUCT_PERCENT_TEXT",
                        "medium",
                        "Tax rate appears to come from a product description percentage, not from tax/MwSt/VAT context.",
                        rate=rate,
                        source_line_ids=tax.get("source_line_ids"),
                        source_text=source_line_text_for_ids(
                            ocr_context, tax.get("source_line_ids") or []
                        ),
                    )
                )
            else:
                issues.append(
                    issue("UNUSUAL_TAX_RATE", "low", "Unusual tax rate extracted.", rate=rate)
                )
        if grand_total is not None and tax_value is not None and abs(tax_value) > abs(grand_total):
            issues.append(
                issue(
                    "TAX_GREATER_THAN_TOTAL",
                    "high",
                    "Tax amount is larger than grand total.",
                    tax=tax_value,
                    grand_total=grand_total,
                )
            )
        if gross is not None and grand_total is not None and abs(gross) > abs(grand_total) + 0.05:
            issues.append(
                issue(
                    "TAX_GROSS_GREATER_THAN_TOTAL",
                    "medium",
                    "Tax gross amount exceeds receipt total.",
                    gross=gross,
                    grand_total=grand_total,
                )
            )

    idx = line_index(ocr_context)
    if not ocr_context.get("layout_rows"):
        issues.append(
            issue(
                "MISSING_LAYOUT_CONTEXT",
                "medium",
                "V14.6 layout/grouped evidence rows are missing from OCR context; LLM had weaker evidence.",
            )
        )
    used_ids = collect_source_line_ids(receipt)
    expanded_used_ids = sorted(expand_used_line_ids_via_layout_rows(used_ids, ocr_context))
    invalid = sorted({lid for lid in used_ids if lid not in idx})
    if invalid:
        issues.append(
            issue(
                "INVALID_EVIDENCE_LINE_IDS",
                "medium",
                "Some source_line_ids do not exist in OCR context.",
                line_ids=invalid[:20],
            )
        )
    counts = Counter(used_ids)
    duplicates = sorted([lid for lid, count in counts.items() if count > 3])
    if duplicates:
        issues.append(
            issue(
                "HEAVILY_REUSED_EVIDENCE_LINE",
                "low",
                "Some OCR lines are reused many times as evidence.",
                line_ids=duplicates[:20],
            )
        )

    low_evidence = []
    for lid in sorted(set(expanded_used_ids)):
        line = idx.get(lid)
        if not line:
            continue
        try:
            conf = float(line.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        if conf and conf < 0.70:
            low_evidence.append(
                {"line_id": lid, "confidence": round(conf, 4), "text": line.get("text")}
            )
    if low_evidence:
        issues.append(
            issue(
                "LOW_CONFIDENCE_USED_EVIDENCE",
                "medium",
                "Some extracted fields rely on low-confidence OCR lines.",
                lines=low_evidence[:20],
            )
        )

    unresolved_amounts = likely_unresolved_amount_lines(receipt, ocr_context)
    if unresolved_amounts:
        # If the core accounting is already balanced, unresolved amount rows are
        # usually duplicate total/payment/tax/context evidence rather than a
        # blocking parse failure. Keep the diagnostic, but do not promote it to a
        # medium issue unless the receipt is numerically unbalanced.
        is_core_balanced = (
            grand_total is not None
            and priced_item_count > 0
            and (difference is None or abs(float(difference or 0.0)) <= tolerance)
        )
        severity = (
            "low" if is_core_balanced else ("medium" if len(unresolved_amounts) >= 3 else "low")
        )
        issues.append(
            issue(
                "UNRESOLVED_AMOUNT_LINES",
                severity,
                "OCR contains amount-like rows not used by the LLM and not listed as unresolved.",
                lines=unresolved_amounts[:30],
                count=len(unresolved_amounts),
            )
        )

    high_or_critical = [i for i in issues if i.get("severity") in {"critical", "high"}]
    balanced = (
        not high_or_critical
        and grand_total is not None
        and priced_item_count > 0
        and (difference is None or abs(difference) <= tolerance)
    )
    if parse_status == "failed":
        import_decision = "llm_failed"
    elif balanced and not [i for i in issues if i.get("severity") == "medium"]:
        import_decision = "import"
    elif not high_or_critical and grand_total is not None:
        import_decision = "needs_review"
    else:
        import_decision = "reject"

    report = {
        "schema_version": "v14_6_validation_report_1",
        "balanced": balanced,
        "import_decision": import_decision,
        "tolerance": tolerance,
        "calculated_item_total": calculated_items,
        "stated_total": grand_total,
        "difference": difference,
        "payment_sum": payment_amount_for_check,
        "payment_count": payment_count_for_report,
        "tax_sum": t_sum,
        "tax_count": tax_count,
        "tax_table_evidence_sum": table_tax_sum,
        "tax_table_evidence_count": table_tax_count,
        "positive_item_total": positive_items_total,
        "discount_item_total": discount_items_total,
        "item_count": len(receipt.get("items") or []),
        "priced_item_count": priced_item_count,
        "missing_price_item_count": missing_price_count,
        "subtotal": subtotal,
        "tax_total": tax_total,
        "issues": issues,
    }
    report["failure_diagnosis"] = diagnose_failure(parse_status, issues, ocr_context)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a V14 LLM-main receipt result")
    parser.add_argument("receipt", type=Path)
    parser.add_argument("ocr_context", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()

    receipt = load_json(args.receipt)
    context = load_json(args.ocr_context)
    report = validate_receipt(receipt, context, tolerance=args.tolerance)
    save_json(args.out, report)
    print(
        json.dumps(
            {
                "balanced": report["balanced"],
                "import_decision": report["import_decision"],
                "difference": report["difference"],
                "issue_count": len(report["issues"]),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["import_decision"] in {"import", "needs_review"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
