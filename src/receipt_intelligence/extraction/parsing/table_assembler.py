#!/usr/bin/env python3
"""Authoritative assembly helpers for LLM table-interpretation evidence.

The assembler uses row-level hybrid arbitration.  Earlier versions switched globally
from VLM table rows to OCR layout rows when a VLM row-shift was detected.  That
fixed shifted top rows but could lose later rows where VLM was better.  The new
strategy is:

- use OCR layout rows where the VLM table is demonstrably shifted;
- deduplicate overlapping OCR candidates;
- attach quantity/unit-price note rows to nearby/related items;
- add VLM-only or OCR-adjustment candidates only when total reconciliation
  supports them;
- keep the table interpreter as semantic evidence, not as a deterministic
  receipt-specific parser.
"""

from __future__ import annotations

import copy
import itertools
import re
from typing import Any

TABLE_ASSEMBLER_SCHEMA_VERSION = "v14_20_row_level_table_assembler_1"
AMOUNT_TOL = 0.03

_NON_ITEM_PAYMENT_RE = re.compile(
    r"\b(RÜCKG|RUECKG|RÜCKGELD|RUECKGELD|CHANGE|BARGELD|CASH|BAR|EC|GIROCARD|KARTE|CARD|GEG\.?|GEGEBEN|ZU\s+BEZAHLEN|ZU\s+ZAHLEN)\b",
    re.IGNORECASE,
)
_QTY_NOTE_RE = re.compile(
    r"^\s*\d+[,.]?\d*\s*(?:STK|STÜCK|STUECK|PCS?|QTY|KG|G|L|ML|PACK|PK)?\s*(?:x|×|@|à|a)\s*[-+]?\d*(?:[,\.]\d+)?\s*$",
    re.IGNORECASE,
)
_DEPOSIT_RE = re.compile(r"\b(PFAND|LEERGUT|LEERG\.?|MEHRWEG|EINWEG)\b", re.IGNORECASE)
_GENERIC_DEPOSIT_RE = re.compile(r"^\s*(PFAND|PFAND\s+EUR|LEERG\.?\s*MW\s*V\.?)\s*$", re.IGNORECASE)


def _amount(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = str(value).strip().replace("−", "-")
    if not text:
        return None
    negative = (
        text.startswith("-")
        or text.endswith("-")
        or bool(re.search(r"[,\.]\s*\d{1,2}\s*[-−]", text))
    )
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


def _confidence(value: Any, default: float = 0.75) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _norm(text: Any) -> str:
    text = str(text or "").upper()
    text = text.replace("Ä", "AE").replace("Ö", "OE").replace("Ü", "UE").replace("ß", "SS")
    text = re.sub(r"\b(EUR|EURO|A|B|STK|STUECK|STÜCK|X)\b", " ", text)
    text = re.sub(r"[-+]?\d+[,.]?\d*", " ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _row_index_from_id(row_id: Any) -> int | None:
    m = re.search(r"(\d+)$", str(row_id or ""))
    return int(m.group(1)) if m else None


def _source_line_ids_from_row(row: dict[str, Any]) -> list[str]:
    ids = row.get("source_line_ids")
    if isinstance(ids, list):
        return [str(x) for x in ids if str(x).startswith("line_")]
    # Do not put VLM/table row ids into source_line_ids because the validator
    # treats this field as OCR line ids. Preserve VLM ids separately in
    # table_interpretation_source_row_id.
    return []


def _is_quantity_note(row: dict[str, Any]) -> bool:
    desc = " ".join(
        str(row.get(k) or "") for k in ("product_description", "description", "raw_description")
    )
    if _QTY_NOTE_RE.match(desc.strip()):
        return True
    return str(row.get("row_type") or "").lower() == "quantity_note"


def _item_kind(desc: str, line_total: float | None, row_type: str = "item") -> str:
    rt = (row_type or "item").lower()
    if rt in {"discount", "deposit", "refund"}:
        return rt
    if _DEPOSIT_RE.search(desc or ""):
        return "refund" if (line_total or 0) < 0 else "deposit"
    return "item"


def _normalize_item_from_row(row: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    line_total = _amount(row.get("line_total"))
    if line_total is None:
        return None
    row_type = str(row.get("row_type") or "item").lower()
    if row_type in {
        "header",
        "subtotal",
        "total",
        "payment",
        "change",
        "tax",
        "note",
        "unknown",
        "quantity_note",
    }:
        return None
    raw_text = " ".join(
        str(row.get(k) or "")
        for k in ("raw_description", "description", "product_description", "raw_cells")
    )
    if _NON_ITEM_PAYMENT_RE.search(raw_text):
        return None
    if _is_quantity_note(row):
        return None
    desc = _text(
        row.get("product_description") or row.get("description") or row.get("raw_description")
    )
    product_desc = _text(row.get("product_description") or desc)
    if not desc and not product_desc:
        return None
    category = _item_kind(desc or product_desc or "", line_total, row_type)
    return {
        "raw_description": _text(row.get("raw_description") or desc),
        "description": desc or product_desc or "unknown item",
        "product_description": product_desc or desc,
        "line_note": _text(row.get("line_note")),
        "promotion_note": _text(row.get("promotion_note")),
        "quantity": _amount(row.get("quantity")),
        "unit": _text(row.get("unit")),
        "unit_price": _amount(row.get("unit_price")),
        "original_price": _amount(row.get("original_price"))
        if row.get("original_price") is not None
        else _amount(row.get("unit_price")),
        "discount_amount": _amount(row.get("discount_amount")),
        "line_total": line_total,
        "tax_rate": _amount(row.get("tax_rate")),
        "tax_code": _text(row.get("tax_code")),
        "category": category,
        "source_line_ids": _source_line_ids_from_row(row),
        "table_interpretation_source_row_id": _text(
            row.get("source_row_id") or row.get("id") or row.get("row_id")
        ),
        "confidence": _confidence(row.get("confidence"), 0.78),
        "notes": _text(row.get("notes")),
        "_source": source,
        "_row_order": _row_index_from_id(
            row.get("source_row_id") or row.get("id") or row.get("row_id")
        ),
    }


def _items_from_table_interpretation(
    table_interpretation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(table_interpretation, dict):
        return items
    for table in table_interpretation.get("tables") or []:
        if not isinstance(table, dict):
            continue
        table_id = str(table.get("source_table_id") or table.get("id") or "table")
        for row in table.get("rows") or []:
            if not isinstance(row, dict):
                continue
            item = _normalize_item_from_row(row, source=table_id)
            if item is not None:
                items.append(item)
    return items


def _ocr_item_from_candidate(cand: dict[str, Any]) -> dict[str, Any] | None:
    desc = _text(cand.get("description"))
    line_total = _amount(cand.get("line_total"))
    if not desc or line_total is None:
        return None
    category = _item_kind(desc, line_total)
    row_id = _text(cand.get("row_id") or cand.get("candidate_id"))
    return {
        "raw_description": cand.get("evidence_text") or desc,
        "description": desc,
        "product_description": desc,
        "line_note": None,
        "promotion_note": None,
        "quantity": None,
        "unit": None,
        "unit_price": None,
        "original_price": None,
        "discount_amount": None,
        "line_total": line_total,
        "tax_rate": None,
        "tax_code": _text(cand.get("tax_code")),
        "category": category,
        "source_line_ids": [str(x) for x in (cand.get("source_line_ids") or []) if x is not None],
        "table_interpretation_source_row_id": row_id,
        "confidence": 0.74,
        "notes": "Item sourced from OCR/VLM row-level arbitration.",
        "_source": "ocr_layout_candidate",
        "_row_order": _row_index_from_id(cand.get("row_id")),
    }


def _candidate_quality(item: dict[str, Any]) -> tuple[int, int, int, int]:
    desc = str(item.get("description") or "")
    norm = _norm(desc)
    source_line_count = len(item.get("source_line_ids") or [])
    non_generic = 0 if _GENERIC_DEPOSIT_RE.match(desc) else 1
    alpha_len = len(re.sub(r"[^A-ZÄÖÜa-zäöüß]", "", desc))
    has_product_word = 1 if len(norm) >= 4 else 0
    return (source_line_count, non_generic, has_product_word, alpha_len)


def _dedupe_ocr_items(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for item in items:
        line_total = _amount(item.get("line_total"))
        source_lines = set(item.get("source_line_ids") or [])
        desc_norm = _norm(item.get("description"))
        replace_idx = None
        duplicate_idx = None
        for idx, old in enumerate(kept):
            old_total = _amount(old.get("line_total"))
            old_lines = set(old.get("source_line_ids") or [])
            if (
                line_total is not None
                and old_total is not None
                and abs(line_total - old_total) <= AMOUNT_TOL
            ):
                overlap = bool(source_lines and old_lines and source_lines.intersection(old_lines))
                same_norm = desc_norm and desc_norm == _norm(old.get("description"))
                generic_pair = bool(
                    _DEPOSIT_RE.search(str(item.get("description") or ""))
                    and _DEPOSIT_RE.search(str(old.get("description") or ""))
                )
                if overlap or same_norm or (generic_pair and overlap):
                    duplicate_idx = idx
                    if _candidate_quality(item) > _candidate_quality(old):
                        replace_idx = idx
                    break
        if duplicate_idx is None:
            kept.append(item)
        elif replace_idx is not None:
            removed = kept[replace_idx]
            kept[replace_idx] = item
            actions.append(
                {
                    "action": "dedupe_replace",
                    "kept": item.get("description"),
                    "removed": removed.get("description"),
                    "amount": line_total,
                }
            )
        else:
            actions.append(
                {
                    "action": "dedupe_drop",
                    "kept": kept[duplicate_idx].get("description"),
                    "removed": item.get("description"),
                    "amount": line_total,
                }
            )
    return kept, actions


def _attach_quantity_notes(
    items: list[dict[str, Any]], arbitration: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not isinstance(arbitration, dict):
        return []
    actions: list[dict[str, Any]] = []
    by_row: dict[str, dict[str, Any]] = {
        str(i.get("table_interpretation_source_row_id")): i
        for i in items
        if i.get("table_interpretation_source_row_id")
    }
    for note in arbitration.get("quantity_note_candidates") or []:
        if not isinstance(note, dict):
            continue
        qty = _amount(note.get("quantity"))
        unit_price = _amount(note.get("unit_price"))
        computed = _amount(note.get("computed_total"))
        if qty is None or unit_price is None:
            continue
        target: dict[str, Any] | None = None
        for match in note.get("matching_item_rows") or []:
            rid = str(match.get("row_id") or "")
            if rid in by_row:
                target = by_row[rid]
                break
        if target is None and computed is not None and unit_price >= 0:
            # Quantity notes for refunds often print a positive unit price while
            # the item line is negative.  Match by absolute total as well.
            # Negative unit-price rows are usually separate refund/adjustment
            # candidates and must not be attached to unrelated deposit rows.
            for item in items:
                lt = _amount(item.get("line_total"))
                if lt is None:
                    continue
                if abs(abs(lt) - abs(computed)) <= AMOUNT_TOL and _DEPOSIT_RE.search(
                    str(item.get("description") or "")
                ):
                    target = item
                    break
        if target is None:
            continue
        target["quantity"] = qty
        target["unit_price"] = abs(unit_price) if abs(float(unit_price)) > 0 else unit_price
        if target.get("original_price") is None:
            target["original_price"] = target["unit_price"]
        text = _text(note.get("evidence_text"))
        if text:
            target["line_note"] = text
        actions.append(
            {
                "action": "attach_quantity_note",
                "item": target.get("description"),
                "quantity": qty,
                "unit_price": unit_price,
                "computed_total": computed,
            }
        )
    return actions


def _same_item_by_text(a: dict[str, Any], b: dict[str, Any]) -> bool:
    na = _norm(a.get("description") or a.get("product_description"))
    nb = _norm(b.get("description") or b.get("product_description"))
    if not na or not nb:
        return False
    if na == nb:
        return True
    # For noisy OCR, allow one normalized description to contain the other when
    # the shorter side is still meaningful.
    return (len(na) >= 5 and na in nb) or (len(nb) >= 5 and nb in na)


def _line_total_counts(items: list[dict[str, Any]]) -> dict[float, int]:
    counts: dict[float, int] = {}
    for item in items:
        lt = _amount(item.get("line_total"))
        if lt is not None:
            counts[round(lt, 2)] = counts.get(round(lt, 2), 0) + 1
    return counts


def _make_supplement_candidates(
    *,
    base_items: list[dict[str, Any]],
    table_items: list[dict[str, Any]],
    arbitration: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    base_amount_counts = _line_total_counts(base_items)
    used_amount_counts: dict[float, int] = {}
    for table_item in table_items:
        lt = _amount(table_item.get("line_total"))
        if lt is None:
            continue
        if any(_same_item_by_text(table_item, base) for base in base_items):
            continue
        # In row-shift cases, many wrong VLM descriptions have amounts already
        # present in OCR. Treat those amounts as already covered by OCR rather
        # than adding shifted duplicate rows.
        covered_by_amount = used_amount_counts.get(round(lt, 2), 0) < base_amount_counts.get(
            round(lt, 2), 0
        )
        if covered_by_amount:
            used_amount_counts[round(lt, 2)] = used_amount_counts.get(round(lt, 2), 0) + 1
            continue
        cand = copy.deepcopy(table_item)
        cand["notes"] = (
            "Supplemental VLM table row considered by total reconciliation because OCR layout missed or under-read it."
        )
        cand["confidence"] = min(float(cand.get("confidence") or 0.72), 0.72)
        cand["_supplement_source"] = "table_only"
        candidates.append(cand)

    if isinstance(arbitration, dict):
        for note in arbitration.get("quantity_note_candidates") or []:
            if not isinstance(note, dict):
                continue
            unit_price = _amount(note.get("unit_price"))
            computed = _amount(note.get("computed_total"))
            if unit_price is None:
                continue
            if note.get("matching_item_rows"):
                continue
            # Negative amount rows that were mis-detected as quantity notes are
            # useful item/refund candidates when they reconcile the printed total.
            # Positive unmatched quantity notes are not emitted as items.
            if unit_price >= 0:
                continue
            desc = "Deposit/refund adjustment"
            # Prefer a concrete negative deposit/refund label from VLM table
            # candidates. This gives rows such as -0.15 a useful label like
            # LEERG. MW V. ST instead of a generic or unrelated PFAND label.
            negative_deposit_labels = [
                str(t.get("description") or "")
                for t in table_items
                if _DEPOSIT_RE.search(str(t.get("description") or ""))
                and (_amount(t.get("line_total")) or 0) < 0
            ]
            if negative_deposit_labels:
                desc = negative_deposit_labels[-1]
            candidates.append(
                {
                    "raw_description": note.get("evidence_text") or desc,
                    "description": desc,
                    "product_description": desc,
                    "line_note": note.get("evidence_text"),
                    "promotion_note": None,
                    "quantity": None,
                    "unit": None,
                    "unit_price": None,
                    "original_price": None,
                    "discount_amount": None,
                    "line_total": unit_price,
                    "tax_rate": None,
                    "tax_code": None,
                    "category": "refund",
                    "source_line_ids": [
                        str(x) for x in (note.get("source_line_ids") or []) if x is not None
                    ],
                    "table_interpretation_source_row_id": _text(
                        note.get("row_id") or note.get("candidate_id")
                    ),
                    "confidence": 0.55,
                    "notes": "Negative orphan amount recovered from OCR quantity-note evidence because total reconciliation required it.",
                    "_source": "negative_orphan_adjustment",
                    "_row_order": _row_index_from_id(note.get("row_id")),
                    "_supplement_source": "negative_orphan_adjustment",
                }
            )
    return candidates


def _choose_reconciliation_supplements(
    *,
    base_items: list[dict[str, Any]],
    supplement_candidates: list[dict[str, Any]],
    target_total: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_total is None or not supplement_candidates:
        return [], {"attempted": False, "reason": "missing_target_or_candidates"}
    base_sum = round(sum(float(i.get("line_total") or 0.0) for i in base_items), 2)
    diff = round(float(target_total) - base_sum, 2)
    if abs(diff) <= AMOUNT_TOL:
        return [], {
            "attempted": True,
            "reason": "base_already_balanced",
            "base_sum": base_sum,
            "target_total": target_total,
            "diff": diff,
        }
    cands = [c for c in supplement_candidates if _amount(c.get("line_total")) is not None][:14]
    best_subset: list[dict[str, Any]] = []
    best_err = abs(diff)
    best_score: tuple[int, int, float] | None = None
    # Try small subsets first.  Receipt mismatches are usually one or two rows.
    for r in range(1, min(5, len(cands)) + 1):
        for combo in itertools.combinations(cands, r):
            s = round(sum(float(_amount(c.get("line_total")) or 0.0) for c in combo), 2)
            err = abs(round(diff - s, 2))
            if err <= AMOUNT_TOL:
                # Prefer fewer rows, concrete table rows, then larger confidence.
                concrete = sum(1 for c in combo if c.get("_supplement_source") == "table_only")
                conf = sum(float(c.get("confidence") or 0.0) for c in combo)
                score = (-r, concrete, conf)
                if best_score is None or score > best_score:
                    best_subset = [copy.deepcopy(c) for c in combo]
                    best_err = err
                    best_score = score
        if best_subset:
            break
    if best_subset:
        return best_subset, {
            "attempted": True,
            "status": "matched",
            "base_sum": base_sum,
            "target_total": target_total,
            "diff": diff,
            "selected_sum": round(sum(float(i.get("line_total") or 0.0) for i in best_subset), 2),
            "selected_count": len(best_subset),
            "residual_error": best_err,
        }
    return [], {
        "attempted": True,
        "status": "no_exact_subset",
        "base_sum": base_sum,
        "target_total": target_total,
        "diff": diff,
        "candidate_count": len(cands),
        "best_residual_error": best_err,
    }


def _strip_private_fields(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        clean = {k: v for k, v in item.items() if not k.startswith("_")}
        out.append(clean)
    return out


def _items_from_arbitration_hybrid(
    table_interpretation: dict[str, Any] | None,
    arbitration: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(arbitration, dict):
        return [], {"source": "none", "reason": "no_arbitration"}
    warnings = arbitration.get("warnings") or []
    has_row_shift = any(
        isinstance(w, dict) and w.get("code") == "VLM_TABLE_POSSIBLE_ROW_SHIFT" for w in warnings
    )
    if not has_row_shift:
        return [], {"source": "none", "reason": "no_row_shift_warning"}

    table_items = _items_from_table_interpretation(table_interpretation)
    ocr_items_raw = [
        _ocr_item_from_candidate(c)
        for c in (arbitration.get("ocr_layout_item_candidates") or [])
        if isinstance(c, dict)
    ]
    ocr_items = [i for i in ocr_items_raw if i is not None]
    base_items, dedupe_actions = _dedupe_ocr_items(ocr_items)
    qty_actions = _attach_quantity_notes(base_items, arbitration)
    target_total = _best_printed_total(table_interpretation, table_items)
    supplement_candidates = _make_supplement_candidates(
        base_items=base_items, table_items=table_items, arbitration=arbitration
    )
    selected_supplements, recon_report = _choose_reconciliation_supplements(
        base_items=base_items,
        supplement_candidates=supplement_candidates,
        target_total=target_total,
    )
    for item in selected_supplements:
        item["notes"] = (
            str(item.get("notes") or "") + " Selected by row-level reconciliation."
        ).strip()
    merged = base_items + selected_supplements
    merged.sort(
        key=lambda i: (
            _row_index_from_id(i.get("table_interpretation_source_row_id"))
            if _row_index_from_id(i.get("table_interpretation_source_row_id")) is not None
            else 9999
        )
    )
    final_sum = round(sum(float(i.get("line_total") or 0.0) for i in merged), 2)
    return _strip_private_fields(merged), {
        "source": "table_arbitration_row_level_hybrid",
        "item_count": len(merged),
        "table_item_count": len(table_items),
        "ocr_item_count_raw": len(ocr_items),
        "ocr_item_count_deduped": len(base_items),
        "target_total": target_total,
        "final_sum": final_sum,
        "final_difference": round(final_sum - float(target_total), 2)
        if target_total is not None
        else None,
        "dedupe_actions": dedupe_actions,
        "quantity_note_actions": qty_actions,
        "supplement_candidate_count": len(supplement_candidates),
        "selected_supplement_count": len(selected_supplements),
        "selected_supplements": [
            {
                "description": i.get("description"),
                "line_total": i.get("line_total"),
                "source": i.get("_supplement_source"),
                "row_id": i.get("table_interpretation_source_row_id"),
            }
            for i in selected_supplements
        ],
        "reconciliation": recon_report,
        "reason": "VLM row-shift was detected, so OCR and VLM evidence were fused per row instead of choosing one source globally.",
    }


def authoritative_items(
    table_interpretation: dict[str, Any] | None,
    arbitration: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the best authoritative item list plus diagnostics."""
    hybrid_items, hybrid_diag = _items_from_arbitration_hybrid(table_interpretation, arbitration)
    if hybrid_items:
        return hybrid_items, hybrid_diag
    table_items = _strip_private_fields(_items_from_table_interpretation(table_interpretation))
    return table_items, {
        "source": "table_interpretation",
        "item_count": len(table_items),
        "table_item_count": len(table_items),
    }


def _best_printed_total(
    table_interpretation: dict[str, Any] | None, items: list[dict[str, Any]] | None = None
) -> float | None:
    if isinstance(table_interpretation, dict):
        settlement = (
            table_interpretation.get("settlement")
            if isinstance(table_interpretation.get("settlement"), dict)
            else {}
        )
        due = _amount(settlement.get("amount_due"))
        if due is not None:
            return due
        for table in table_interpretation.get("tables") or []:
            if not isinstance(table, dict):
                continue
            val = _amount(table.get("printed_total_candidate"))
            if val is not None:
                return val
    if items:
        return round(sum(float(i.get("line_total") or 0.0) for i in items), 2)
    return None


def _settlement_payments(
    table_interpretation: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    if not isinstance(table_interpretation, dict):
        return [], None, None
    settlement = (
        table_interpretation.get("settlement")
        if isinstance(table_interpretation.get("settlement"), dict)
        else {}
    )
    payments: list[dict[str, Any]] = []
    paid_total = None
    for tender in settlement.get("tenders") or []:
        if not isinstance(tender, dict):
            continue
        typ = str(tender.get("type") or "unknown").lower().replace(" ", "_")
        amount = _amount(tender.get("amount"))
        if amount is None:
            continue
        sign_meaning = str(tender.get("sign_meaning") or "").lower()
        if typ in {"refund", "change"} or sign_meaning == "returned_to_customer":
            continue
        payments.append(
            {
                "method": typ,
                "amount": abs(amount),
                "source_line_ids": [str(tender.get("source_row_id"))]
                if tender.get("source_row_id")
                else [],
                "raw_label": tender.get("raw_label"),
                "sign_meaning": tender.get("sign_meaning"),
            }
        )
    if payments:
        paid_total = round(sum(float(p.get("amount") or 0.0) for p in payments), 2)
    change = _amount(settlement.get("change_due_to_customer"))
    return payments, paid_total, change


def assemble_receipt_from_table_interpretation(
    *,
    table_interpretation: dict[str, Any] | None,
    arbitration: dict[str, Any] | None = None,
    base_receipt: dict[str, Any] | None = None,
    reason: str = "assembled_from_llm_table_interpretation",
) -> dict[str, Any]:
    """Build a valid receipt object from authoritative LLM table evidence."""
    base = copy.deepcopy(base_receipt) if isinstance(base_receipt, dict) else {}
    items, item_diag = authoritative_items(table_interpretation, arbitration)
    grand_total = _best_printed_total(table_interpretation, items)
    payments, paid_total, change = _settlement_payments(table_interpretation)

    merchant = base.get("merchant") if isinstance(base.get("merchant"), dict) else {}
    totals_base = base.get("totals") if isinstance(base.get("totals"), dict) else {}
    receipt = {
        "schema_version": "v14_6_llm_receipt_1",
        "parse_status": "partial",
        "currency": base.get("currency") or "EUR",
        "merchant": {
            "name": merchant.get("name"),
            "address": merchant.get("address"),
            "tax_id": merchant.get("tax_id"),
            "source_line_ids": merchant.get("source_line_ids") or [],
        },
        "date": base.get("date"),
        "time": base.get("time"),
        "items": items,
        "taxes": base.get("taxes") if isinstance(base.get("taxes"), list) else [],
        "totals": {
            "subtotal": totals_base.get("subtotal")
            if totals_base.get("subtotal") is not None
            else grand_total,
            "tax_total": totals_base.get("tax_total"),
            "grand_total": totals_base.get("grand_total")
            if totals_base.get("grand_total") is not None
            else grand_total,
            "paid_total": totals_base.get("paid_total")
            if totals_base.get("paid_total") is not None
            else paid_total,
            "change": totals_base.get("change")
            if totals_base.get("change") is not None
            else change,
            "source_line_ids": totals_base.get("source_line_ids") or [],
        },
        "payments": base.get("payments")
        if isinstance(base.get("payments"), list) and base.get("payments")
        else payments,
        "unresolved_rows": base.get("unresolved_rows")
        if isinstance(base.get("unresolved_rows"), list)
        else [],
        "warnings": [
            *(str(x) for x in (base.get("warnings") or []) if str(x).strip()),
            f"Receipt item table was {reason}; main parser may have been skipped/failed or item rows were replaced to avoid re-extraction.",
        ],
        "overall_confidence": min(
            0.85, max(0.55, float((table_interpretation or {}).get("overall_confidence") or 0.65))
        ),
        "assembly": {
            "schema_version": TABLE_ASSEMBLER_SCHEMA_VERSION,
            "reason": reason,
            "item_source": item_diag,
            "grand_total_source": "settlement/table_printed_total/sum_items",
        },
    }
    return receipt


def merge_authoritative_table_items(
    receipt: dict[str, Any],
    table_interpretation: dict[str, Any] | None,
    arbitration: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace weak/failed item extraction with authoritative table items."""
    if not isinstance(receipt, dict):
        return receipt, {"changed": False, "reason": "receipt_not_object"}
    items, diag = authoritative_items(table_interpretation, arbitration)
    if not items:
        return receipt, {"changed": False, "reason": "no_authoritative_items", **diag}
    old_items = receipt.get("items") if isinstance(receipt.get("items"), list) else []
    old_priced = len(
        [i for i in old_items if isinstance(i, dict) and _amount(i.get("line_total")) is not None]
    )
    prefer_sources = {
        "table_arbitration_row_level_hybrid",
        "table_arbitration_ocr_layout_candidates",
    }
    prefer = (
        diag.get("source") in prefer_sources
        or old_priced == 0
        or len(items) >= max(old_priced + 2, 1)
    )
    if not prefer:
        return receipt, {
            "changed": False,
            "reason": "existing_items_not_weaker",
            "old_priced": old_priced,
            **diag,
        }
    out = copy.deepcopy(receipt)
    out["items"] = items
    out.setdefault("warnings", [])
    if isinstance(out["warnings"], list):
        out["warnings"].append(
            "Item rows replaced with authoritative LLM table interpretation/row-level arbitration output."
        )
    totals = out.get("totals") if isinstance(out.get("totals"), dict) else {}
    if totals.get("grand_total") is None:
        totals["grand_total"] = _best_printed_total(table_interpretation, items)
    if totals.get("subtotal") is None:
        totals["subtotal"] = totals.get("grand_total")
    out["totals"] = totals
    return out, {"changed": True, "old_priced": old_priced, **diag}


def compact_visual_evidence_for_main_parser(
    visual_evidence: dict[str, Any] | None,
    *,
    table_interpretation: dict[str, Any] | None = None,
    arbitration: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Shrink VLM evidence before the final LLM parser."""
    if not isinstance(visual_evidence, dict):
        return visual_evidence
    summary = (
        visual_evidence.get("summary") if isinstance(visual_evidence.get("summary"), dict) else {}
    )
    compact: dict[str, Any] = {
        "schema_version": visual_evidence.get("schema_version"),
        "status": visual_evidence.get("status"),
        "backend": visual_evidence.get("backend"),
        "runner": visual_evidence.get("runner"),
        "engine": visual_evidence.get("engine"),
        "device": visual_evidence.get("device"),
        "summary": summary,
        "semantic_guidance": [
            "Dedicated table interpretation is authoritative for item rows when present.",
            "If row-level arbitration is present, use its hybrid guidance instead of globally trusting either VLM or OCR.",
            "Do not re-extract the full item table; focus on merchant/date/totals/payments/tax/unresolved evidence.",
        ],
    }
    if isinstance(table_interpretation, dict):
        slim_ti = copy.deepcopy(table_interpretation)
        for table in slim_ti.get("tables") or []:
            if not isinstance(table, dict):
                continue
            slim_rows = []
            for row in (table.get("rows") or [])[:70]:
                if not isinstance(row, dict):
                    continue
                slim_rows.append(
                    {
                        k: row.get(k)
                        for k in (
                            "source_row_id",
                            "row_type",
                            "product_description",
                            "line_note",
                            "promotion_note",
                            "quantity",
                            "unit",
                            "unit_price",
                            "original_price",
                            "discount_amount",
                            "line_total",
                            "tax_code",
                            "linked_to_row_id",
                            "confidence",
                        )
                    }
                )
            source_table_id = table.get("source_table_id")
            table_type = table.get("table_type")
            confidence = table.get("confidence")
            sum_line_total = table.get("sum_line_total")
            printed_total_candidate = table.get("printed_total_candidate")
            difference = table.get("difference")
            table.clear()
            table.update(
                {
                    "source_table_id": source_table_id,
                    "table_type": table_type,
                    "confidence": confidence,
                    "sum_line_total": sum_line_total,
                    "printed_total_candidate": printed_total_candidate,
                    "difference": difference,
                    "rows": slim_rows,
                }
            )
        compact["table_interpretation"] = {
            "status": slim_ti.get("status"),
            "overall_confidence": slim_ti.get("overall_confidence"),
            "tables": slim_ti.get("tables") or [],
            "settlement": slim_ti.get("settlement")
            if isinstance(slim_ti.get("settlement"), dict)
            else {},
            "warnings": (slim_ti.get("warnings") or [])[:6],
        }
    if isinstance(arbitration, dict):
        warnings = (arbitration.get("warnings") or [])[:8]
        has_row_shift = any(
            isinstance(w, dict) and w.get("code") == "VLM_TABLE_POSSIBLE_ROW_SHIFT"
            for w in warnings
        )
        arb_small = {
            "summary": arbitration.get("summary"),
            "warnings": warnings,
            "guidance": arbitration.get("guidance") or [],
        }
        if has_row_shift:

            def slim_item(c: Any) -> dict[str, Any]:
                if not isinstance(c, dict):
                    return {}
                return {
                    "id": c.get("candidate_id"),
                    "row_id": c.get("row_id"),
                    "description": c.get("description"),
                    "line_total": c.get("line_total"),
                    "tax_code": c.get("tax_code"),
                    "source_line_ids": c.get("source_line_ids") or [],
                    "product_percent_not_tax": c.get("product_percent_not_tax"),
                }

            def slim_qty(c: Any) -> dict[str, Any]:
                if not isinstance(c, dict):
                    return {}
                return {
                    "id": c.get("candidate_id"),
                    "row_id": c.get("row_id"),
                    "quantity": c.get("quantity"),
                    "unit_price": c.get("unit_price"),
                    "computed_total": c.get("computed_total"),
                    "matching_item_rows": c.get("matching_item_rows") or [],
                }

            arb_small["ocr_layout_item_candidates"] = [
                slim_item(c) for c in (arbitration.get("ocr_layout_item_candidates") or [])[:24]
            ]
            arb_small["quantity_note_candidates"] = [
                slim_qty(c) for c in (arbitration.get("quantity_note_candidates") or [])[:12]
            ]
            arb_small["product_percent_not_tax_rows"] = (
                arbitration.get("product_percent_not_tax_rows") or []
            )[:5]
        compact["table_arbitration"] = arb_small
    for key, limit in (
        ("payment_change_lines", 18),
        ("total_payment_rows", 18),
        ("total_payment_reconciliation_candidates", 8),
        ("tax_like_lines", 16),
        ("tax_table_candidates", 1),
    ):
        val = visual_evidence.get(key)
        if isinstance(val, list):
            compact[key] = val[:limit]
        elif isinstance(val, dict):
            compact[key] = val
    return compact
