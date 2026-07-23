#!/usr/bin/env python3
"""Validation-gated right-column price recovery for split receipt rows.

This module is deliberately conservative. It does not replace the normal OCR,
VLM, LLM or table-arbitration pipeline. It only activates after validation has
already shown an item-sum mismatch.  It uses bounded right-column re-OCR evidence
and OCR/VLM arbitration candidates to propose small add/replace operations that
move the receipt closer to the printed total.

The intent is to handle receipts where product labels are visible on the left
but the right-side price column was split, missed or associated with the wrong
label.  Recovered rows are marked as review-required and should not be imported
into analytics/RAG without human review.
"""

from __future__ import annotations

import copy
import itertools
import re
from typing import Any

from receipt_intelligence.extraction.repair.item_order import sort_items_by_printed_order

SCHEMA_VERSION = "v14_21_right_column_recovery_1"
AMOUNT_TOL = 0.03

PRODUCT_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")
NON_PRODUCT_RE = re.compile(
    r"\b(SUMME|TOTAL|GESAMT|BONSUMME|MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|BAR|CASH|GEGEBEN|R[ÜU]CK|RUECK|CHANGE|KARTE|EC|GIROCARD|VISA|MASTERCARD|BELEG|BON|DATUM|UHRZEIT|TERMINAL|AS-ZEIT|BETRAG|ZAHLUNG|TELEFON|FAX|UID|STRASSE|DÜSSELDORF|DUESSELDORF)\b",
    re.I,
)
UNIT_PRICE_HINT_RE = re.compile(r"\b(EUR\s*/\s*KG|€/KG|/KG|KG\s*x|STK\s*[x×@àa])\b", re.I)


def _amount(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = str(value).strip().replace("−", "-")
    if not text:
        return None
    negative = text.startswith("-") or text.endswith("-")
    cleaned = re.sub(r"[^0-9,\.\s]", "", text).replace(" ", "")
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        val = round(float(cleaned), 2)
    except Exception:
        return None
    return -abs(val) if negative else val


def _norm(text: Any) -> str:
    text = str(text or "").upper()
    text = text.replace("Ä", "AE").replace("Ö", "OE").replace("Ü", "UE").replace("ß", "SS")
    text = re.sub(r"\b(EUR|EURO|A|B|STK|STUECK|STÜCK|X|KG|G|ML|L)\b", " ", text)
    text = re.sub(r"[-+]?\d+[,.]?\d*%?", " ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _product_like(text: Any) -> bool:
    t = str(text or "").strip()
    if len(t) < 4 or not PRODUCT_WORD_RE.search(t):
        return False
    if NON_PRODUCT_RE.search(t):
        return False
    n = _norm(t)
    if n in {"REWE", "REWE MARKT", "EMN", "EUR"}:
        return False
    return True


def _receipt_total(receipt: dict[str, Any], report: dict[str, Any]) -> float | None:
    totals = receipt.get("totals") if isinstance(receipt.get("totals"), dict) else {}
    for v in (report.get("stated_total"), totals.get("grand_total"), totals.get("subtotal")):
        amt = _amount(v)
        if amt is not None:
            return amt
    return None


def _item_sum(items: list[dict[str, Any]]) -> float:
    return round(
        sum(float(_amount(i.get("line_total")) or 0.0) for i in items if isinstance(i, dict)), 2
    )


def _existing_norms(items: list[dict[str, Any]]) -> set[str]:
    return {
        _norm(i.get("product_description") or i.get("description"))
        for i in items
        if isinstance(i, dict)
    }


def _line_set(item: dict[str, Any]) -> set[str]:
    return {str(x) for x in (item.get("source_line_ids") or []) if str(x).startswith("line_")}


def _candidate_from_arbitration(c: dict[str, Any]) -> dict[str, Any] | None:
    desc = str(c.get("description") or "").strip()
    amt = _amount(c.get("line_total"))
    if not _product_like(desc) or amt is None:
        return None
    return {
        "description": desc,
        "product_description": desc,
        "line_total": amt,
        "tax_code": c.get("tax_code"),
        "source_line_ids": [str(x) for x in (c.get("source_line_ids") or [])],
        "row_id": c.get("row_id") or c.get("candidate_id"),
        "source": "ocr_layout_arbitration",
        "confidence": 0.72,
        "evidence_text": c.get("evidence_text"),
    }


def _replacement_actions(
    receipt_items: list[dict[str, Any]], arbitration: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not isinstance(arbitration, dict):
        return []
    cands = [
        _candidate_from_arbitration(c)
        for c in (arbitration.get("ocr_layout_item_candidates") or [])
        if isinstance(c, dict)
    ]
    cands = [c for c in cands if c is not None]
    existing_names = _existing_norms(receipt_items)
    actions: list[dict[str, Any]] = []
    for idx, item in enumerate(receipt_items):
        if not isinstance(item, dict):
            continue
        item_lines = _line_set(item)
        if not item_lines:
            continue
        item_amt = _amount(item.get("line_total"))
        item_norm = _norm(item.get("product_description") or item.get("description"))
        for cand in cands:
            cand_lines = set(cand.get("source_line_ids") or [])
            cand_norm = _norm(cand.get("description"))
            cand_amt = _amount(cand.get("line_total"))
            if not cand_lines.intersection(item_lines):
                continue
            if cand_amt is None or item_amt is None or abs(cand_amt - item_amt) > AMOUNT_TOL:
                continue
            if cand_norm == item_norm:
                continue
            # If the OCR row-level candidate uses the same lines and same amount,
            # it is usually a better label for those lines than a shifted LLM label.
            # Only apply if the candidate label is not already present elsewhere.
            if cand_norm in existing_names:
                continue
            actions.append(
                {
                    "type": "replace_item_label_same_amount",
                    "item_index": idx,
                    "old_description": item.get("description"),
                    "new_description": cand.get("description"),
                    "line_total": cand_amt,
                    "source_line_ids": sorted(cand_lines),
                    "reason": "Existing item uses the same OCR lines/amount but a different product label; OCR row-level candidate is more likely the true label.",
                }
            )
            existing_names.discard(item_norm)
            existing_names.add(cand_norm)
            break
    return actions


def _candidate_amounts_from_reocr_line(
    row: dict[str, Any], target_total: float | None
) -> list[float]:
    values: list[float] = []
    text = str(row.get("recognized_text") or "")
    for a in row.get("amounts") or []:
        amt = _amount(a.get("value") if isinstance(a, dict) else a)
        if amt is None:
            continue
        if target_total is not None and abs(amt - target_total) <= AMOUNT_TOL:
            continue
        if amt <= 0 or amt > 300:
            continue
        # Product-line recovery should use item totals, not obvious unit prices.
        # Keep kg/unit hints only when the candidate product itself is not already present.
        values.append(amt)
    # Preserve order but remove duplicates.
    out: list[float] = []
    for v in values:
        if not any(abs(v - old) <= AMOUNT_TOL for old in out):
            out.append(v)
    return out[:4]


def _addition_candidates(
    *,
    receipt_items: list[dict[str, Any]],
    reocr_result: dict[str, Any] | None,
    target_total: float | None,
) -> list[dict[str, Any]]:
    if not isinstance(reocr_result, dict):
        return []
    existing = _existing_norms(receipt_items)
    candidates: list[dict[str, Any]] = []
    for row in (reocr_result.get("evidence_lines") or []) + (
        reocr_result.get("crop_results") or []
    ):
        if not isinstance(row, dict):
            continue
        desc = str(row.get("product_text_candidate") or "").strip()
        if not _product_like(desc):
            continue
        norm = _norm(desc)
        if norm in existing:
            continue
        amounts = _candidate_amounts_from_reocr_line(row, target_total)
        # If the crop contains both the product line total and a unit price, try
        # both as alternatives. The later subset-selection step decides using
        # the printed total; no candidate is applied just because it exists.
        for amt in amounts:
            candidates.append(
                {
                    "type": "add_recovered_item",
                    "description": desc,
                    "product_description": desc,
                    "line_total": amt,
                    "tax_code": None,
                    "source_line_ids": [str(x) for x in (row.get("source_line_ids") or [])],
                    "row_id": row.get("layout_row_id") or row.get("id"),
                    "source": "right_column_reocr",
                    "confidence": 0.58,
                    "requires_review": True,
                    "evidence_text": row.get("recognized_text"),
                    "crop_path": row.get("crop_path"),
                }
            )
    # Deduplicate exact desc/amount candidates.
    deduped: list[dict[str, Any]] = []
    keys: set[tuple[str, float]] = set()
    for c in candidates:
        key = (_norm(c.get("description")), round(float(c.get("line_total") or 0.0), 2))
        if key in keys:
            continue
        keys.add(key)
        deduped.append(c)
    return deduped[:18]


def _select_additions(
    base_items: list[dict[str, Any]], additions: list[dict[str, Any]], target_total: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_sum = _item_sum(base_items)
    diff = round(target_total - base_sum, 2)
    if abs(diff) <= AMOUNT_TOL:
        return [], {
            "status": "already_balanced",
            "base_sum": base_sum,
            "target_total": target_total,
            "diff": diff,
        }
    if not additions:
        return [], {
            "status": "no_candidates",
            "base_sum": base_sum,
            "target_total": target_total,
            "diff": diff,
        }
    best: list[dict[str, Any]] = []
    best_err = abs(diff)
    best_sum = 0.0
    usable = additions[:14]
    for r in range(1, min(6, len(usable)) + 1):
        for combo in itertools.combinations(usable, r):
            s = round(sum(float(c.get("line_total") or 0.0) for c in combo), 2)
            err = abs(round(diff - s, 2))
            if err < best_err - 1e-9 or (abs(err - best_err) <= 1e-9 and best and r < len(best)):
                best = [copy.deepcopy(c) for c in combo]
                best_err = err
                best_sum = s
        if best_err <= AMOUNT_TOL:
            break
    return best, {
        "status": "matched"
        if best_err <= AMOUNT_TOL
        else ("improved" if best else "no_improvement"),
        "base_sum": base_sum,
        "target_total": target_total,
        "diff": diff,
        "selected_sum": best_sum,
        "selected_count": len(best),
        "residual_error": round(best_err, 2),
        "candidate_count": len(additions),
    }


def _apply_replacements(
    items: list[dict[str, Any]], replacements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out = copy.deepcopy(items)
    for action in replacements:
        idx = action.get("item_index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(out):
            continue
        item = out[idx]
        new_desc = action.get("new_description")
        if not new_desc:
            continue
        item["raw_description"] = item.get("raw_description") or item.get("description")
        item["description"] = new_desc
        item["product_description"] = new_desc
        item["confidence"] = min(float(item.get("confidence") or 0.7), 0.72)
        item["requires_review"] = True
        note = str(item.get("notes") or "").strip()
        extra = "Product label replaced by right-column recovery because OCR line evidence indicated a shifted label."
        item["notes"] = (note + " " + extra).strip()
        item["recovery_source"] = "right_column_recovery_label_replacement"
    return out


def _addition_to_item(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_description": c.get("description"),
        "description": c.get("description"),
        "product_description": c.get("product_description") or c.get("description"),
        "line_note": "Recovered from right-column re-OCR; requires human review.",
        "promotion_note": None,
        "quantity": None,
        "unit": None,
        "unit_price": None,
        "original_price": None,
        "discount_amount": None,
        "line_total": _amount(c.get("line_total")),
        "tax_rate": None,
        "tax_code": c.get("tax_code"),
        "category": "item",
        "source_line_ids": c.get("source_line_ids") or [],
        "table_interpretation_source_row_id": c.get("row_id"),
        "confidence": c.get("confidence") or 0.58,
        "requires_review": True,
        "notes": "Validation-gated item recovered from right-column price evidence.",
        "recovery_source": c.get("source") or "right_column_recovery",
        "recovery_evidence_text": c.get("evidence_text"),
    }


def run_right_column_recovery(
    *,
    receipt: dict[str, Any],
    validation_report: dict[str, Any],
    ocr_context: dict[str, Any] | None = None,
    reocr_result: dict[str, Any] | None = None,
    table_arbitration: dict[str, Any] | None = None,
    tolerance: float = 0.03,
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "receipt_not_object",
        }
    items = receipt.get("items") if isinstance(receipt.get("items"), list) else []
    target_total = _receipt_total(receipt, validation_report)
    if target_total is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "missing_target_total",
        }
    before_sum = _item_sum(items)
    before_diff = round(target_total - before_sum, 2)
    if abs(before_diff) <= max(float(tolerance), AMOUNT_TOL):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped_balanced",
            "reason": "existing_items_already_match_printed_total",
            "before_sum": before_sum,
            "target_total": target_total,
            "before_diff": before_diff,
        }
    issue_codes = {
        str(i.get("code")) for i in (validation_report.get("issues") or []) if isinstance(i, dict)
    }
    if not issue_codes.intersection({"ITEM_SUM_MISMATCH", "NO_ITEMS", "MISSING_TOTAL"}):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "no_item_sum_trigger",
            "issue_codes": sorted(issue_codes),
        }

    replacements = _replacement_actions(items, table_arbitration)
    replaced_items = _apply_replacements(items, replacements)
    additions = _addition_candidates(
        receipt_items=replaced_items, reocr_result=reocr_result, target_total=target_total
    )
    selected, selection = _select_additions(replaced_items, additions, target_total)
    recovered_items = [_addition_to_item(c) for c in selected]
    candidate_items = sort_items_by_printed_order(replaced_items + recovered_items)
    after_sum = _item_sum(candidate_items)
    after_diff = round(target_total - after_sum, 2)
    improved = abs(after_diff) + 1e-9 < abs(before_diff)
    applied = improved and (replacements or selected)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "applied" if applied else "no_improvement",
        "applied": bool(applied),
        "requires_human_review": bool(applied),
        "before_sum": before_sum,
        "target_total": target_total,
        "before_diff": before_diff,
        "after_sum": after_sum if applied else before_sum,
        "after_diff": after_diff if applied else before_diff,
        "improved": bool(improved),
        "issue_codes": sorted(issue_codes),
        "replacement_actions": replacements,
        "addition_candidates": additions,
        "selected_additions": selected,
        "selection": selection,
        "guidance": [
            "Right-column recovery is validation-gated and does not run on already balanced receipts.",
            "Recovered rows are marked requires_review=true and should not enter analytics/RAG without approval.",
            "The recovery only proposes small add/replace operations that improve reconciliation with the printed total.",
        ],
    }
    if applied:
        out = copy.deepcopy(receipt)
        out["items"] = candidate_items
        out.setdefault("warnings", [])
        if isinstance(out["warnings"], list):
            out["warnings"].append(
                "Right-column recovery changed item rows; human review required before DB/RAG import."
            )
        out["right_column_recovery"] = {
            "applied": True,
            "before_sum": before_sum,
            "after_sum": after_sum,
            "target_total": target_total,
            "after_diff": after_diff,
            "replacement_count": len(replacements),
            "selected_addition_count": len(selected),
        }
        out["overall_confidence"] = min(float(out.get("overall_confidence") or 0.7), 0.72)
        result["receipt"] = out
    return result
