#!/usr/bin/env python3
"""Validation-triggered bounded right-column re-OCR evidence.

This module does not build final receipt semantics. It only creates extra OCR
EVIDENCE for the LLM correction pass when validation indicates that item totals
are below the printed total and some product-like rows have no attached price.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from receipt_intelligence.engines.ocr_engine import _run_paddle_ocr_v13_profile
from receipt_intelligence.extraction.evidence.layout import extract_ocr_amounts

PRODUCT_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")
NON_PRODUCT_RE = re.compile(
    r"\b(SUMME|TOTAL|GESAMT|BONSUMME|MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|BAR|CASH|GEGEBEN|R[ÜUO]CKGELD|RUECKGELD|CHANGE|KARTE|EC|GIROCARD|VISA|MASTERCARD|BELEG|BON|DATUM|UHRZEIT|TERMINAL)\b",
    re.I,
)
FOOTER_RE = re.compile(
    r"\b(SUMME|TOTAL|GESAMT|BONSUMME|ZU\s*ZAHLEN|MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|BAR|CASH|GEGEBEN|R[ÜUO]CKGELD|RUECKGELD|CHANGE)\b",
    re.I,
)
HEADER_OR_SHOP_RE = re.compile(
    r"\b(REWE|MARKT|TELEFON|FAX|UID|UST-ID|DÜSSELDORF|DUESSELDORF|STRASSE|STRAßE|KASSE|BED\.|BELEG|BON|DATUM|UHRZEIT)\b",
    re.I,
)
UNIT_OR_QTY_RE = re.compile(r"\b(kg\s*x|EUR\s*/\s*kg|€/kg|Stk\s*[x×@àa])\b", re.I)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _emit(
    callback: Callable[[dict[str, Any]], None] | None,
    stage: str,
    status: str,
    message: str,
    **details: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(
            {
                "stage": stage,
                "status": status,
                "message": message,
                "details": details,
                "source": "receipt_reocr_repair",
            }
        )
    except Exception:
        pass


def _norm_bbox(line: dict[str, Any]) -> tuple[float, float, float, float] | None:
    b = line.get("bbox") if isinstance(line.get("bbox"), dict) else {}
    try:
        x = float(b.get("x"))
        y = float(b.get("y"))
        w = float(b.get("w"))
        h = float(b.get("h"))
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _line_index(ocr_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(line.get("line_id")): line
        for line in (ocr_context.get("lines") or [])
        if isinstance(line, dict)
    }


def _is_product_like_text(text: str) -> bool:
    if not text or not PRODUCT_WORD_RE.search(text):
        return False
    if NON_PRODUCT_RE.search(text):
        return False
    if len(text.strip()) < 4:
        return False
    return True


def _footer_start_index(rows: list[dict[str, Any]]) -> int:
    for i, row in enumerate(rows):
        text = str(row.get("full_text") or row.get("left_text") or "")
        if FOOTER_RE.search(text):
            return i
    return len(rows)


def _candidate_product_rows(
    ocr_context: dict[str, Any], max_candidates: int
) -> list[dict[str, Any]]:
    """Return product-like rows without attached prices, prioritized for repair.

    Earlier versions simply took the first N unpriced product-like rows.  On
    long receipts that wasted crops on headers/shop/address rows and missed the
    actual failed item rows.  This version limits candidates to the item region
    before SUMME, excludes merchant/footer/unit-note lines, and prioritizes
    rows that sit between priced item rows.
    """
    rows = [r for r in (ocr_context.get("layout_rows") or []) if isinstance(r, dict)]
    idx = _line_index(ocr_context)
    stop = _footer_start_index(rows)

    # Start after the first obvious receipt/table header.  This avoids cropping
    # shop logos/address lines such as REWE, phone/fax/UID, etc.
    start = 0
    for i, row in enumerate(rows[:stop]):
        text = str(row.get("full_text") or row.get("left_text") or "").strip()
        if text.upper() == "EUR":
            start = i + 1
            break

    candidates: list[dict[str, Any]] = []
    for i, row in enumerate(rows[start:stop], start=start):
        if row.get("right_amount_value") is not None:
            continue
        text = str(row.get("left_text") or row.get("full_text") or "").strip()
        if not _is_product_like_text(text):
            continue
        if HEADER_OR_SHOP_RE.search(text) or UNIT_OR_QTY_RE.search(text):
            continue
        source_ids = [str(x) for x in (row.get("source_line_ids") or [])]
        bboxes = [_norm_bbox(idx[s]) for s in source_ids if s in idx]
        bboxes = [b for b in bboxes if b is not None]
        if not bboxes:
            continue
        # Prefer rows that are visually in the product column (left side) and
        # surrounded by priced/amount rows.  This catches split product/price
        # pairs while keeping the routine generic.
        min_x = min(b[0] for b in bboxes)
        if min_x > 0.55:
            continue
        nearby_has_amount = False
        for j in range(max(start, i - 3), min(stop, i + 4)):
            if j == i:
                continue
            if rows[j].get("right_amount_value") is not None:
                nearby_has_amount = True
                break
        candidates.append(
            {
                "layout_row_id": row.get("row_id"),
                "text": text,
                "source_line_ids": source_ids,
                "bboxes": bboxes,
                "layout_index": i,
                "nearby_has_amount": nearby_has_amount,
            }
        )

    candidates.sort(
        key=lambda c: (0 if c.get("nearby_has_amount") else 1, c.get("layout_index") or 9999)
    )
    return candidates[:max_candidates]


def _crop_right_column(
    image_path: Path, cand: dict[str, Any], crop_dir: Path, crop_id: str
) -> Path | None:
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            W, H = img.size
            ys = []
            hs = []
            for _x, y, _w, h in cand.get("bboxes") or []:
                ys.append(y)
                hs.append(h)
            if not ys:
                return None
            y0 = max(0.0, min(ys) - max(hs) * 1.2)
            y1 = min(1.0, max(y + h for _, y, _, h in cand.get("bboxes") or []) + max(hs) * 1.6)
            # Generic receipt assumption: item price is usually in the right half/right column.
            x0 = 0.45
            x1 = 0.995
            box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            crop_dir.mkdir(parents=True, exist_ok=True)
            out = crop_dir / f"{crop_id}.jpg"
            img.crop(box).save(out, quality=95)
            cand["crop_box_norm"] = [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]
            return out
    except Exception:
        return None


def run_bounded_right_column_reocr(
    *,
    image_path: Path | None,
    ocr_context: dict[str, Any],
    validation_report: dict[str, Any],
    result_dir: Path,
    run_id: str,
    enabled: bool = True,
    max_crops: int = 8,
    lang: str = "german",
    device: str = "cpu",
    min_score: float = 0.20,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = result_dir / f"{run_id}_v14_6_right_column_reocr.json"
    issue_codes = {
        str(i.get("code")) for i in (validation_report.get("issues") or []) if isinstance(i, dict)
    }
    if not enabled:
        result = {"status": "disabled", "schema_version": "v14_6_right_column_reocr_1"}
        _save_json(out_path, result)
        return result
    if image_path is None or not Path(image_path).exists():
        result = {
            "status": "skipped",
            "schema_version": "v14_6_right_column_reocr_1",
            "reason": "missing_source_image",
        }
        _save_json(out_path, result)
        return result
    if not issue_codes.intersection({"ITEM_SUM_MISMATCH", "UNRESOLVED_AMOUNT_LINES", "NO_ITEMS"}):
        result = {
            "status": "skipped",
            "schema_version": "v14_6_right_column_reocr_1",
            "reason": "no_item_ocr_repair_trigger",
            "triggered_by_issue_codes": sorted(issue_codes),
        }
        _save_json(out_path, result)
        return result

    cands = _candidate_product_rows(ocr_context, max_candidates=max_crops)
    if not cands:
        result = {
            "status": "skipped",
            "schema_version": "v14_6_right_column_reocr_1",
            "reason": "no_product_like_unpriced_rows",
            "triggered_by_issue_codes": sorted(issue_codes),
        }
        _save_json(out_path, result)
        return result

    _emit(
        progress_callback,
        "right_column_reocr",
        "running",
        "Running bounded right-column re-OCR for product-like rows without attached prices.",
        candidate_count=len(cands),
    )
    crop_dir = result_dir / f"{run_id}_v14_6_reocr_crops"
    evidence_lines: list[dict[str, Any]] = []
    crop_results: list[dict[str, Any]] = []
    for n, cand in enumerate(cands[:max_crops]):
        crop = _crop_right_column(
            Path(image_path), cand, crop_dir, f"crop_{n:03d}_{cand.get('layout_row_id')}"
        )
        if crop is None:
            continue
        try:
            ocr = _run_paddle_ocr_v13_profile(crop, lang=lang, device=device, min_score=min_score)
            texts = [
                str(w.get("text") or "").strip()
                for w in (ocr.get("words") or [])
                if str(w.get("text") or "").strip()
            ]
            joined = " ".join(texts)
            amounts = extract_ocr_amounts(joined)
            item = {
                "id": f"reocr_line_{len(evidence_lines):03d}",
                "product_text_candidate": cand.get("text"),
                "source_line_ids": cand.get("source_line_ids"),
                "layout_row_id": cand.get("layout_row_id"),
                "crop_path": str(crop),
                "crop_box_norm": cand.get("crop_box_norm"),
                "recognized_text": joined,
                "amounts": amounts,
                "tags": ["item_price_like", "right_column_reocr"],
            }
            if amounts:
                evidence_lines.append(item)
            crop_results.append(item)
        except Exception as exc:
            crop_results.append(
                {
                    "layout_row_id": cand.get("layout_row_id"),
                    "product_text_candidate": cand.get("text"),
                    "crop_path": str(crop),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    result = {
        "schema_version": "v14_6_right_column_reocr_1",
        "status": "ok" if evidence_lines else "no_amounts_found",
        "triggered_by_issue_codes": sorted(issue_codes),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "candidate_count": len(cands),
        "evidence_line_count": len(evidence_lines),
        "evidence_lines": evidence_lines,
        "crop_results": crop_results,
        "generic_rule": "Right-column re-OCR is evidence only. Candidate selection is validation-triggered and prioritizes unpriced product rows in the item region; deterministic recovery may only apply candidates when they improve printed-total reconciliation and marks them for review.",
    }
    _save_json(out_path, result)
    _emit(
        progress_callback,
        "right_column_reocr",
        "done" if evidence_lines else "warning",
        "Bounded right-column re-OCR finished.",
        status_detail=result.get("status"),
        evidence_line_count=len(evidence_lines),
    )
    return result


def reocr_evidence_to_visual_evidence(
    reocr_result: dict[str, Any], validation_report: dict[str, Any]
) -> dict[str, Any]:
    lines = []
    for row in reocr_result.get("evidence_lines") or []:
        lines.append(
            {
                "id": row.get("id"),
                "text": f"{row.get('product_text_candidate')} | {row.get('recognized_text')}",
                "amounts": row.get("amounts") or [],
                "tags": row.get("tags") or ["item_price_like", "right_column_reocr"],
                "path": row.get("crop_path"),
            }
        )
    return {
        "schema_version": "v14_6_visual_evidence_from_reocr_1",
        "status": reocr_result.get("status"),
        "backend": "bounded_right_column_reocr",
        "triggered_by_issue_codes": [
            str(i.get("code"))
            for i in (validation_report.get("issues") or [])
            if isinstance(i, dict)
        ],
        "summary": {
            "line_count": len(lines),
            "amount_line_count": len([line for line in lines if line.get("amounts")]),
            "has_payment_like": False,
            "has_change_like": False,
            "has_tax_like": False,
        },
        "lines": lines,
        "amount_lines": lines,
        "payment_change_lines": [],
        "tax_like_lines": [],
        "item_price_like_lines": lines,
        "engine_error": None,
    }
