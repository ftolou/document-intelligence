#!/usr/bin/env python3
"""Vertical price-stack recovery for split product/price receipt rows.

This recovery layer is intentionally validation-gated. It is only attempted for
unbalanced receipts and it never changes already balanced receipts.  The goal is
not to replace the normal OCR/VLM/LLM pipeline.  It creates a second, full-column
item-table candidate when the normal row pairing lost right-side prices.

Typical failure mode:

    PRODUCT A                 1,99 B
    PRODUCT B                 2,99 B

is seen by OCR/layout as separate left product rows and an incomplete/shifted
right amount column.  Per-row crops often fail because the crop is too small.
This module crops the whole right-side price band in the item region, OCRs the
vertical stack, extracts ordered amount tokens with y-positions, pairs them with
left-side product rows by y/order, and applies the candidate only if it improves
printed-total reconciliation.  Any applied result is marked review-required.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from receipt_intelligence.engines.ocr_engine import _run_paddle_ocr_v13_profile
from receipt_intelligence.extraction.evidence.layout import extract_ocr_amounts
from receipt_intelligence.extraction.repair.item_order import (
    sort_items_by_printed_order,
    sort_records_by_source_position,
)

SCHEMA_VERSION = "v14_22_vertical_price_stack_recovery_3"
AMOUNT_TOL = 0.03

PRODUCT_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")
NON_PRODUCT_RE = re.compile(
    r"\b(SUMME|TOTAL|GESAMT|BONSUMME|ZU\s*ZAHLEN|MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|BAR|CASH|GEGEBEN|R[ÜU]CK|RUECK|CHANGE|KARTE|EC|GIROCARD|VISA|MASTERCARD|BELEG|BON|DATUM|UHRZEIT|TERMINAL|AS-ZEIT|BETRAG|ZAHLUNG|TELEFON|FAX|UID|STRASSE|DÜSSELDORF|DUESSELDORF)\b",
    re.I,
)
FOOTER_RE = re.compile(
    r"\b(SUMME|TOTAL|GESAMT|BONSUMME|ZU\s*ZAHLEN|MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|BAR|CASH|GEGEBEN|R[ÜUO]CKGELD|RUECKGELD|CHANGE|BETRAG)\b",
    re.I,
)
HEADER_RE = re.compile(
    r"\b(TELEFON|FAX|UID|UST-ID|DÜSSELDORF|DUESSELDORF|STRASSE|STRAßE|KASSE|BED\.|BELEG|BON|DATUM|UHRZEIT)\b",
    re.I,
)
UNIT_OR_QTY_RE = re.compile(
    r"\b(kg\s*x|EUR\s*/\s*kg|€/kg|/kg|Stk\s*[x×@àa]|\d+[,.]?\d*\s*(kg|g|ml|l)\s*[x×])\b", re.I
)
CURRENCY_ONLY_RE = re.compile(r"^\s*(EUR|EURO|€)\s*$", re.I)
TAX_CODE_RE = re.compile(r"\b([AB])\b", re.I)


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
                "source": SCHEMA_VERSION,
            }
        )
    except Exception:
        pass


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
        or re.search(r",\s*\d{2}\s*[-−]", text) is not None
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


def _compact_digits(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _normalize_price_text(text: Any, *, allow_digit_only: bool = True) -> list[dict[str, Any]]:
    """Extract amount candidates from noisy price-column OCR text.

    PaddleOCR on narrow crops can split decimals (`1, 99`), attach VAT letters
    (`4,59A`), or return digit-only strings (`459`).  This function is more
    permissive than the normal receipt amount extractor, but the caller still
    applies arithmetic validation before accepting any candidate.
    """
    raw = str(text or "").strip()
    if not raw:
        return []
    # Remove currency and isolated tax markers while keeping numeric structure.
    cleaned = raw.replace("−", "-").replace("€", " ")
    cleaned = re.sub(r"\b(EUR|EURO)\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=\d)\s*[AB]\b", " ", cleaned, flags=re.I)
    cleaned = cleaned.replace("O", "0").replace("o", "0")
    out: list[dict[str, Any]] = []

    def add(value: float | None, candidate_raw: str, quality: str) -> None:
        if value is None or abs(value) < 0.01 or abs(value) > 300:
            return
        value = round(float(value), 2)
        if any(
            abs(value - float(existing["value"])) <= AMOUNT_TOL
            and candidate_raw == existing.get("raw")
            for existing in out
        ):
            return
        out.append({"value": value, "raw": candidate_raw.strip(), "quality": quality})

    # First use the existing strict extractor.  It handles common German formats.
    try:
        for cand in extract_ocr_amounts(raw):
            add(_amount(cand.get("value")), str(cand.get("raw") or raw), "strict")
    except Exception:
        pass

    # Decimal comma/dot with optional spaces: 1,99 / 1, 99 / 1 . 99.
    for m in re.finditer(r"[-+]?\d{1,3}\s*[,\.]\s*\d{2}(?!\d)", cleaned):
        text_m = m.group(0)
        norm = _compact_digits(text_m).replace(",", ".")
        add(_amount(norm), text_m, "spaced_decimal")

    # Digit-only parsing is intentionally narrow.  It is useful when OCR returns
    # `459` for `4,59`, but dangerous when the same row already contains a
    # decimal candidate plus noise, e.g. `1, 58 8`.  In that case the previous
    # implementation also parsed `588 -> 5.88`, which caused a false recovery.
    if allow_digit_only and not out and not re.search(r"[,\.]", cleaned):
        compact = _compact_digits(cleaned)
        if re.fullmatch(r"[-+]?\d{3,5}", compact):
            tok = compact
            sign = -1 if tok.startswith("-") else 1
            tok = tok.lstrip("+-")
            val = sign * float(f"{tok[:-2]}.{tok[-2:]}")
            add(val, compact, "digit_only_price_column")

    return out


def _ocr_word_bbox(word: dict[str, Any]) -> tuple[float, float, float, float] | None:
    b = word.get("bbox") if isinstance(word.get("bbox"), dict) else {}
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


def _cluster_ocr_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster OCR word/line boxes into y-ordered text rows.

    PaddleOCR may return line-level boxes for the full image and word/fragment
    boxes for the crop.  This normalizes both cases into rows so split fragments
    like `1` + `99` can be reconstructed as one amount.
    """
    prepared: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        b = _ocr_word_bbox(row)
        if not b:
            continue
        x, y, w, h = b
        prepared.append(
            {
                "idx": idx,
                "text": text,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cy": y + h / 2.0,
                "cx": x + w / 2.0,
            }
        )
    if not prepared:
        return []
    heights = sorted(r["h"] for r in prepared if r["h"] > 0)
    median_h = heights[len(heights) // 2] if heights else 0.025
    y_tol = max(0.012, min(0.045, median_h * 0.85))

    clusters: list[list[dict[str, Any]]] = []
    for r in sorted(prepared, key=lambda z: (z["cy"], z["x"])):
        target = None
        for c in clusters:
            c_cy = sum(float(x["cy"]) for x in c) / max(len(c), 1)
            if abs(float(r["cy"]) - c_cy) <= y_tol:
                target = c
                break
        if target is None:
            clusters.append([r])
        else:
            target.append(r)

    out: list[dict[str, Any]] = []
    for ci, cluster in enumerate(clusters):
        cluster = sorted(cluster, key=lambda z: z["x"])
        text = " ".join(str(z["text"]) for z in cluster).strip()
        x0 = min(float(z["x"]) for z in cluster)
        y0 = min(float(z["y"]) for z in cluster)
        x1 = max(float(z["x"] + z["w"]) for z in cluster)
        y1 = max(float(z["y"] + z["h"]) for z in cluster)
        out.append(
            {
                "row_index": ci,
                "text": text,
                "bbox": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                "word_indices": [z["idx"] for z in cluster],
            }
        )
    return sorted(
        out,
        key=lambda r: (
            float((r.get("bbox") or {}).get("y", 0.0)),
            float((r.get("bbox") or {}).get("x", 0.0)),
        ),
    )


def _amount_tokens_from_rows(
    rows: list[dict[str, Any]],
    *,
    bounds: dict[str, float],
    crop_path: Path | None,
    variant: str,
    coordinate_space: str,
) -> list[dict[str, Any]]:
    x0 = float(bounds["x0"])
    y0 = float(bounds["y0"])
    xspan = float(bounds["x1"]) - x0
    yspan = float(bounds["y1"]) - y0
    out: list[dict[str, Any]] = []
    for n, row in enumerate(rows):
        text = str(row.get("text") or "").strip()
        if not text or CURRENCY_ONLY_RE.match(text):
            continue
        candidates = _normalize_price_text(text, allow_digit_only=True)
        if not candidates:
            continue
        b = row.get("bbox") if isinstance(row.get("bbox"), dict) else {}
        try:
            cy_local = float(b.get("y", 0.0)) + float(b.get("h", 0.0)) / 2.0
            cx_local = float(b.get("x", 0.0)) + float(b.get("w", 0.0)) / 2.0
        except Exception:
            cy_local = 0.0
            cx_local = 0.0
        if coordinate_space == "crop":
            cy_orig = y0 + cy_local * yspan
            cx_orig = x0 + cx_local * xspan
        else:
            cy_orig = cy_local
            cx_orig = cx_local
        for cand in candidates:
            val = _amount(cand.get("value"))
            if val is None:
                continue
            out.append(
                {
                    "id": f"stack_amount_{len(out):03d}",
                    "value": val,
                    "raw": cand.get("raw"),
                    "parse_quality": cand.get("quality"),
                    "text": text,
                    "tax_code": _extract_tax_code(text),
                    "y_center": round(cy_orig, 5),
                    "x_center": round(cx_orig, 5),
                    "crop_path": str(crop_path) if crop_path else None,
                    "crop_variant": variant,
                    "ocr_row_index": n,
                    "coordinate_space": coordinate_space,
                }
            )
    out.sort(
        key=lambda r: (
            float(r.get("y_center") or 0.0),
            float(r.get("x_center") or 0.0),
            float(r.get("value") or 0.0),
        )
    )
    return _dedupe_amount_tokens(out)


def _dedupe_amount_tokens(
    tokens: list[dict[str, Any]], *, y_tol: float = 0.007
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for r in sorted(
        tokens, key=lambda z: (float(z.get("y_center") or 0.0), float(z.get("x_center") or 0.0))
    ):
        if any(
            abs(float(r.get("y_center") or 0.0) - float(old.get("y_center") or 0.0)) <= y_tol
            and abs(float(r.get("value") or 0.0) - float(old.get("value") or 0.0)) <= AMOUNT_TOL
            for old in deduped
        ):
            continue
        r = dict(r)
        r["id"] = f"stack_amount_{len(deduped):03d}"
        deduped.append(r)
    return deduped


def _amount_tokens_from_ocr_context(
    ocr_context: dict[str, Any], *, bounds: dict[str, float]
) -> list[dict[str, Any]]:
    """Fallback amount extraction from the original full-image OCR boxes.

    The crop OCR can fail on narrow columns.  Full-image OCR often already saw
    some right-column amounts, so we include those as a secondary evidence set.
    """
    x0 = float(bounds["x0"])
    x1 = float(bounds["x1"])
    y0 = float(bounds["y0"])
    y1 = float(bounds["y1"])
    rows: list[dict[str, Any]] = []
    for line in ocr_context.get("lines") or []:
        if not isinstance(line, dict):
            continue
        b = _bbox(line)
        if not b:
            continue
        lx, ly, lw, lh = b
        cy = ly + lh / 2.0
        cx = lx + lw / 2.0
        if cy < y0 or cy > y1 or cx < x0 or cx > x1:
            continue
        text = str(line.get("text") or "")
        if CURRENCY_ONLY_RE.match(text) or FOOTER_RE.search(text):
            continue
        rows.append({"text": text, "bbox": {"x": cx, "y": cy, "w": 0.001, "h": 0.001}})
    return _amount_tokens_from_rows(
        rows,
        bounds=bounds,
        crop_path=None,
        variant="full_image_right_column",
        coordinate_space="original",
    )


def _merge_amount_evidence(*sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for source in sets:
        for token in source or []:
            merged.append(dict(token))
    # Prefer strong parses if two values occupy the same y neighborhood.
    quality_rank = {"strict": 0, "spaced_decimal": 1, "digit_only_price_column": 2}
    merged.sort(
        key=lambda r: (
            float(r.get("y_center") or 0.0),
            quality_rank.get(str(r.get("parse_quality")), 9),
        )
    )
    return _dedupe_amount_tokens(merged, y_tol=0.010)


def _preprocess_price_crop_variants(crop: Image.Image, crop_dir: Path) -> list[tuple[str, Path]]:
    """Create OCR-friendly crop variants for tiny receipt amount stacks."""
    variants: list[tuple[str, Image.Image]] = []
    base = crop.convert("RGB")
    variants.append(("1x", base))
    variants.append(
        (
            "2x",
            base.resize(
                (max(1, base.width * 2), max(1, base.height * 2)), Image.Resampling.LANCZOS
            ),
        )
    )
    # Widening via padding gives the detector more context around isolated digits.
    padded = ImageOps.expand(base, border=max(8, base.width // 12), fill="white")
    variants.append(
        (
            "3x_padded",
            padded.resize(
                (max(1, padded.width * 3), max(1, padded.height * 3)), Image.Resampling.LANCZOS
            ),
        )
    )
    gray = ImageOps.grayscale(padded)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)
    variants.append(
        (
            "4x_contrast",
            gray.resize(
                (max(1, gray.width * 4), max(1, gray.height * 4)), Image.Resampling.LANCZOS
            ).convert("RGB"),
        )
    )
    # Binary variant: sometimes PaddleOCR recognizes separated decimal commas better
    # after background cleanup.  Keep both; validation picks the best evidence.
    binary = gray.point(lambda px: 0 if px < 175 else 255, mode="1").convert("RGB")
    variants.append(
        (
            "4x_binary",
            binary.resize(
                (max(1, binary.width * 4), max(1, binary.height * 4)), Image.Resampling.LANCZOS
            ),
        )
    )

    paths: list[tuple[str, Path]] = []
    for label, image in variants:
        path = crop_dir / f"price_stack_column_{label}.jpg"
        image.save(path, quality=95)
        paths.append((label, path))
    # Backward-compatible canonical names used by earlier docs/UI.
    (crop_dir / "price_stack_column.jpg").write_bytes(
        (crop_dir / "price_stack_column_1x.jpg").read_bytes()
    )
    (crop_dir / "price_stack_column_2x.jpg").write_bytes(
        (crop_dir / "price_stack_column_2x.jpg").read_bytes()
    )
    return paths


def _norm(text: Any) -> str:
    text = str(text or "").upper()
    text = text.replace("Ä", "AE").replace("Ö", "OE").replace("Ü", "UE").replace("ß", "SS")
    text = re.sub(r"\b(EUR|EURO|A|B|STK|STUECK|STÜCK|X|KG|G|ML|L)\b", " ", text)
    text = re.sub(r"[-+]?\d+[,.]?\d*%?", " ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _product_like(text: Any) -> bool:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(t) < 4 or not PRODUCT_WORD_RE.search(t):
        return False
    if (
        CURRENCY_ONLY_RE.match(t)
        or HEADER_RE.search(t)
        or NON_PRODUCT_RE.search(t)
        or UNIT_OR_QTY_RE.search(t)
    ):
        return False
    n = _norm(t)
    if not n or n in {"REWE", "REWE MARKT", "EMN", "EUR"}:
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


def _line_index(ocr_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(l.get("line_id")): l for l in (ocr_context.get("lines") or []) if isinstance(l, dict)
    }


def _bbox(line: dict[str, Any]) -> tuple[float, float, float, float] | None:
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


def _center_y_from_lines(row: dict[str, Any], idx: dict[str, dict[str, Any]]) -> float | None:
    ys: list[float] = []
    for sid in row.get("source_line_ids") or []:
        line = idx.get(str(sid))
        if not line:
            continue
        b = _bbox(line)
        if b:
            ys.append(b[1] + b[3] / 2.0)
    if not ys:
        return None
    return sum(ys) / len(ys)


def _row_bbox(
    row: dict[str, Any], idx: dict[str, dict[str, Any]]
) -> tuple[float, float, float, float] | None:
    bboxes = []
    for sid in row.get("source_line_ids") or []:
        line = idx.get(str(sid))
        b = _bbox(line) if line else None
        if b:
            bboxes.append(b)
    if not bboxes:
        return None
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[0] + b[2] for b in bboxes)
    y1 = max(b[1] + b[3] for b in bboxes)
    return x0, y0, x1 - x0, y1 - y0


def _item_region_rows(ocr_context: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    rows = [r for r in (ocr_context.get("layout_rows") or []) if isinstance(r, dict)]
    stop = len(rows)
    for i, row in enumerate(rows):
        text = str(row.get("full_text") or row.get("left_text") or "")
        if FOOTER_RE.search(text):
            stop = i
            break
    start = 0
    for i, row in enumerate(rows[:stop]):
        text = str(row.get("full_text") or row.get("left_text") or "").strip()
        if CURRENCY_ONLY_RE.match(text):
            start = i + 1
            break
    return rows, start, stop


def _product_rows(ocr_context: dict[str, Any]) -> list[dict[str, Any]]:
    rows, start, stop = _item_region_rows(ocr_context)
    idx = _line_index(ocr_context)
    products: list[dict[str, Any]] = []
    for i, row in enumerate(rows[start:stop], start=start):
        text = str(row.get("left_text") or row.get("full_text") or "").strip()
        # Remove the right-side value if layout row represented it as text after a pipe.
        if "|" in text:
            text = text.split("|", 1)[0].strip()
        if not _product_like(text):
            continue
        rb = _row_bbox(row, idx)
        cy = _center_y_from_lines(row, idx)
        if rb is None or cy is None:
            continue
        # Product text should primarily be in the left/middle part of the receipt.
        if rb[0] > 0.58:
            continue
        products.append(
            {
                "row_id": row.get("row_id"),
                "layout_index": i,
                "description": text,
                "product_description": text,
                "source_line_ids": [str(x) for x in (row.get("source_line_ids") or [])],
                "y_center": round(cy, 5),
                "bbox_norm": [
                    round(rb[0], 5),
                    round(rb[1], 5),
                    round(rb[0] + rb[2], 5),
                    round(rb[1] + rb[3], 5),
                ],
            }
        )
    # Deduplicate product rows by normalized text and nearby y.
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for p in products:
        key = (_norm(p.get("description")), int(float(p.get("y_center") or 0.0) * 1000))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _right_column_bounds(
    ocr_context: dict[str, Any], products: list[dict[str, Any]]
) -> dict[str, float] | None:
    if not products:
        return None
    rows, start, stop = _item_region_rows(ocr_context)
    idx = _line_index(ocr_context)
    y0 = max(0.0, min(float(p["y_center"]) for p in products) - 0.018)
    # Stop just before the first footer/total row, otherwise the crop may pick up SUMME/payment values.
    y1_candidates: list[float] = []
    if stop < len(rows):
        rb = _row_bbox(rows[stop], idx)
        if rb:
            y1_candidates.append(rb[1] - 0.004)
    y1_candidates.append(min(0.99, max(float(p["y_center"]) for p in products) + 0.035))
    # Use the earliest plausible end boundary: if a SUMME/footer row is found,
    # do not include it in the item price stack crop. Still guarantee a minimum
    # crop height so small receipts do not collapse.
    y1 = max(y0 + 0.05, min(0.99, min(y1_candidates)))

    # Estimate the right amount column from amount-like OCR lines in the item region.
    xs: list[float] = []
    for line in ocr_context.get("lines") or []:
        if not isinstance(line, dict):
            continue
        b = _bbox(line)
        if not b:
            continue
        cy = b[1] + b[3] / 2.0
        if cy < y0 or cy > y1:
            continue
        text = str(line.get("text") or "")
        amounts = extract_ocr_amounts(text)
        if b[0] > 0.58 and amounts:
            xs.append(b[0])
            xs.append(b[0] + b[2])
    if xs:
        # Use a deliberately wider band than the detected amount boxes.  Narrow
        # price crops are fragile because the detector loses decimal commas and
        # tax letters at the edges.  Keep the left boundary above ~0.64 so unit
        # prices in the middle column (EUR/kg, Stk x) are not pulled in.
        x0 = max(0.64, min(xs) - 0.075)
        x1 = min(0.99, max(xs) + 0.055)
    else:
        # Generic fallback for German supermarket receipts: final item totals are usually in the far right band.
        x0 = 0.66
        x1 = 0.99
    return {"x0": round(x0, 5), "y0": round(y0, 5), "x1": round(x1, 5), "y1": round(y1, 5)}


def _crop_and_ocr_stack(
    *,
    image_path: Path,
    bounds: dict[str, float],
    result_dir: Path,
    run_id: str,
    lang: str,
    device: str,
    min_score: float,
) -> dict[str, Any]:
    crop_dir = result_dir / f"{run_id}_v14_22_vertical_price_stack_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    x0 = float(bounds["x0"])
    y0 = float(bounds["y0"])
    x1 = float(bounds["x1"])
    y1 = float(bounds["y1"])
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        W, H = img.size
        box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
        crop = img.crop(box)
        variant_paths = _preprocess_price_crop_variants(crop, crop_dir)

    # OCR all variants.  Use a lower min_score than the full-image pass because
    # price-column crops often produce low-confidence but still useful digits.
    ocr_min_score = min(float(min_score), 0.08)
    variants: list[dict[str, Any]] = []
    for label, path in variant_paths:
        try:
            ocr = _run_paddle_ocr_v13_profile(
                path, lang=lang, device=device, min_score=ocr_min_score
            )
            amounts = _amount_tokens_from_crop_ocr(
                ocr, bounds=bounds, crop_path=path, variant=label
            )
            rows_for_text = _cluster_ocr_rows((ocr.get("lines") or []) + (ocr.get("words") or []))
            variants.append(
                {
                    "variant": label,
                    "crop_path": str(path),
                    "ocr_word_count": len(ocr.get("words") or []),
                    "ocr_line_count": len(ocr.get("lines") or []),
                    "recognized_text": "\n".join(
                        str(r.get("text") or "") for r in rows_for_text if isinstance(r, dict)
                    ),
                    "amount_count": len(amounts),
                    "amount_sum": round(sum(float(a.get("value") or 0.0) for a in amounts), 2),
                    "amounts": amounts,
                }
            )
        except Exception as exc:
            variants.append(
                {
                    "variant": label,
                    "crop_path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "amounts": [],
                    "amount_count": 0,
                    "amount_sum": 0.0,
                }
            )

    merged_amounts = _merge_amount_evidence(*[v.get("amounts") or [] for v in variants])
    best = (
        max(
            variants,
            key=lambda v: (
                int(v.get("amount_count") or 0),
                float(v.get("amount_sum") or 0.0),
                int(v.get("ocr_word_count") or 0),
            ),
        )
        if variants
        else {"amounts": []}
    )
    return {
        "bounds_norm": bounds,
        "crop_variants": variants,
        "best_variant": best.get("variant"),
        "amount_count": len(merged_amounts),
        "amount_sum": round(sum(float(a.get("value") or 0.0) for a in merged_amounts), 2),
        "amounts": merged_amounts,
    }


def _amount_tokens_from_crop_ocr(
    ocr: dict[str, Any], *, bounds: dict[str, float], crop_path: Path, variant: str
) -> list[dict[str, Any]]:
    # Use both line and word boxes.  Some PaddleOCR versions return crop text in
    # `words` only; others return line boxes.  Clustering allows fragments like
    # `1` + `99` to form one candidate row.
    raw_rows: list[dict[str, Any]] = []
    for key in ("lines", "words"):
        raw_rows.extend([r for r in (ocr.get(key) or []) if isinstance(r, dict)])
    clustered = _cluster_ocr_rows(raw_rows)

    # Also parse raw boxes individually so a clean full amount line is not lost
    # by clustering with a neighboring tax marker or noise fragment.
    individual = [r for r in raw_rows if isinstance(r, dict)]
    rows = clustered + individual
    return _amount_tokens_from_rows(
        rows, bounds=bounds, crop_path=crop_path, variant=variant, coordinate_space="crop"
    )


def _extract_tax_code(text: str) -> str | None:
    m = TAX_CODE_RE.search(text or "")
    if not m:
        return None
    return m.group(1).lower()


def _candidate_items_by_y(
    products: list[dict[str, Any]], amounts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not products or not amounts:
        return [], {
            "status": "no_products_or_amounts",
            "product_count": len(products),
            "amount_count": len(amounts),
        }
    products_sorted = sorted(products, key=lambda p: float(p.get("y_center") or 0.0))
    amounts_sorted = sorted(amounts, key=lambda a: float(a.get("y_center") or 0.0))

    # Estimate row spacing from product rows. This allows small vertical offsets
    # from OCR where price appears slightly above/below the product text baseline.
    gaps: list[float] = []
    for a, b in zip(products_sorted, products_sorted[1:]):
        gap = float(b.get("y_center") or 0.0) - float(a.get("y_center") or 0.0)
        if gap > 0:
            gaps.append(gap)
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.02
    max_dist = max(0.018, min(0.045, median_gap * 1.15))

    paired: list[dict[str, Any]] = []
    used_amounts: set[int] = set()
    for p in products_sorted:
        py = float(p.get("y_center") or 0.0)
        best_i = None
        best_dist = 999.0
        for i, a in enumerate(amounts_sorted):
            if i in used_amounts:
                continue
            ay = float(a.get("y_center") or 0.0)
            dist = abs(ay - py)
            if dist < best_dist:
                best_i = i
                best_dist = dist
        if best_i is not None and best_dist <= max_dist:
            a = amounts_sorted[best_i]
            used_amounts.add(best_i)
            paired.append(_product_amount_to_item(p, a, match_mode="nearest_y", distance=best_dist))

    # If many rows remain unpaired but counts are close, use sequence fallback.
    # This is useful when OCR detects the stack with consistent vertical order
    # but y coordinates are systematically shifted by one narrow line.
    if len(paired) < min(len(products_sorted), len(amounts_sorted)) * 0.70:
        paired = []
        used_amounts = set()
        for p, a in zip(products_sorted, amounts_sorted):
            paired.append(
                _product_amount_to_item(
                    p,
                    a,
                    match_mode="ordered_stack",
                    distance=abs(float(a.get("y_center") or 0.0) - float(p.get("y_center") or 0.0)),
                )
            )
    return paired, {
        "status": "ok" if paired else "no_pairs",
        "product_count": len(products_sorted),
        "amount_count": len(amounts_sorted),
        "paired_count": len(paired),
        "median_product_gap": round(median_gap, 5),
        "max_pair_distance": round(max_dist, 5),
        "used_amount_count": len(used_amounts)
        if used_amounts
        else min(len(products_sorted), len(amounts_sorted))
        if paired
        else 0,
    }


def _product_amount_to_item(
    p: dict[str, Any], a: dict[str, Any], *, match_mode: str, distance: float
) -> dict[str, Any]:
    desc = str(p.get("description") or "").strip()
    return {
        "raw_description": desc,
        "description": desc,
        "product_description": desc,
        "line_note": "Recovered by vertical price-stack OCR; requires human review.",
        "promotion_note": None,
        "quantity": None,
        "unit": None,
        "unit_price": None,
        "original_price": None,
        "discount_amount": None,
        "line_total": _amount(a.get("value")),
        "tax_rate": None,
        "tax_code": a.get("tax_code"),
        "category": "item",
        "source_line_ids": p.get("source_line_ids") or [],
        "table_interpretation_source_row_id": p.get("row_id"),
        "confidence": 0.62,
        "requires_review": True,
        "notes": "Validation-gated item row reconstructed from left product text and full right-side vertical price stack.",
        "recovery_source": "vertical_price_stack_recovery",
        "recovery_match_mode": match_mode,
        "recovery_y_distance": round(float(distance), 5),
        "recovery_amount_raw": a.get("raw"),
        "recovery_amount_text": a.get("text"),
        "recovery_product_y_center": p.get("y_center"),
        "recovery_layout_index": p.get("layout_index"),
        "recovery_amount_y_center": a.get("y_center"),
    }


def _existing_names(items: list[dict[str, Any]]) -> set[str]:
    return {
        _norm(i.get("product_description") or i.get("description"))
        for i in items
        if isinstance(i, dict)
    }


def _full_stack_candidate(
    receipt: dict[str, Any],
    candidate_items: list[dict[str, Any]],
    target_total: float,
    before_diff: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return a replacement candidate only when it balances to the printed total.

    Earlier versions allowed partial improvements.  That was unsafe: a receipt
    could move from `-12.92` difference to `-7.04` while adding a wrong row such
    as `KART. VORW. FESTK. = 5.88`.  This layer is now candidate-only unless the
    full reconstructed table reconciles to the printed total within tolerance.
    """
    if not candidate_items:
        return None, {"status": "no_candidate_items"}
    candidate_items = sort_items_by_printed_order(
        candidate_items,
        sequences=[candidate_items, receipt.get("items") or []],
    )
    cand_sum = _item_sum(candidate_items)
    cand_diff = round(target_total - cand_sum, 2)
    matched = abs(cand_diff) <= AMOUNT_TOL
    if not matched:
        return None, {
            "status": "candidate_only_not_balanced",
            "candidate_sum": cand_sum,
            "candidate_diff": cand_diff,
            "before_diff": before_diff,
            "policy": "no_partial_vertical_stack_mutation",
        }
    out = copy.deepcopy(receipt)
    out["items"] = copy.deepcopy(candidate_items)
    out.setdefault("warnings", [])
    if isinstance(out["warnings"], list):
        out["warnings"].append(
            "Vertical price-stack recovery replaced item rows; human review required before DB/RAG import."
        )
    out["vertical_price_stack_recovery"] = {
        "applied": True,
        "mode": "full_stack_item_reconstruction",
        "candidate_sum": cand_sum,
        "target_total": target_total,
        "candidate_diff": cand_diff,
        "item_count": len(candidate_items),
        "policy": "balanced_only",
    }
    out["overall_confidence"] = min(float(out.get("overall_confidence") or 0.7), 0.68)
    return out, {
        "status": "matched",
        "candidate_sum": cand_sum,
        "candidate_diff": cand_diff,
        "candidate_count": len(candidate_items),
        "existing_count": len(receipt.get("items") or []),
    }


def _addition_candidate_mode(
    receipt: dict[str, Any],
    candidate_items: list[dict[str, Any]],
    target_total: float,
    before_diff: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Candidate-only additive fallback.

    Additive recovery is now also balanced-only.  A partial improvement is saved
    in the artifact diagnostics, but it must not mutate the final receipt.
    """
    existing = _existing_names(
        receipt.get("items") if isinstance(receipt.get("items"), list) else []
    )
    missing = [
        i
        for i in candidate_items
        if _norm(i.get("product_description") or i.get("description")) not in existing
    ]
    if not missing:
        return None, {"status": "no_missing_item_candidates"}
    best: list[dict[str, Any]] = []
    best_err = abs(before_diff)
    best_sum = 0.0
    from itertools import combinations

    usable = missing[:14]
    for r in range(1, min(7, len(usable)) + 1):
        for combo in combinations(usable, r):
            s = round(sum(float(_amount(i.get("line_total")) or 0.0) for i in combo), 2)
            err = abs(round(before_diff - s, 2))
            if err < best_err - 1e-9:
                best = [copy.deepcopy(i) for i in combo]
                best_err = err
                best_sum = s
        if best_err <= AMOUNT_TOL:
            break
    if not best:
        return None, {
            "status": "no_improving_addition_subset",
            "missing_candidate_count": len(missing),
            "residual_error": round(best_err, 2),
        }
    if best_err > AMOUNT_TOL:
        return None, {
            "status": "candidate_only_partial_addition_not_applied",
            "selected_count": len(best),
            "selected_sum": best_sum,
            "residual_error": round(best_err, 2),
            "missing_candidate_count": len(missing),
            "policy": "no_partial_vertical_stack_mutation",
        }
    out = copy.deepcopy(receipt)
    combined_items = copy.deepcopy(receipt.get("items") or []) + best
    out["items"] = sort_items_by_printed_order(
        combined_items,
        sequences=[candidate_items, receipt.get("items") or []],
    )
    out.setdefault("warnings", [])
    if isinstance(out["warnings"], list):
        out["warnings"].append(
            "Vertical price-stack recovery added item rows; human review required before DB/RAG import."
        )
    out["vertical_price_stack_recovery"] = {
        "applied": True,
        "mode": "add_missing_items",
        "selected_count": len(best),
        "selected_sum": best_sum,
        "policy": "balanced_only",
    }
    return out, {
        "status": "matched",
        "selected_count": len(best),
        "residual_error": round(best_err, 2),
        "missing_candidate_count": len(missing),
    }


def _evidence_item(
    *,
    description: Any,
    amount_value: Any = None,
    amount_raw: Any = None,
    tax_code: Any = None,
    source_line_ids: list[str] | None = None,
    evidence_source: str,
    priority: int,
    confidence: float = 0.70,
) -> dict[str, Any] | None:
    desc = re.sub(r"\s+", " ", str(description or "")).strip()
    if not _product_like(desc):
        return None
    amt = _amount(amount_value)
    return {
        "raw_description": desc,
        "description": desc,
        "product_description": desc,
        "line_note": "Recovered from fused region/table evidence; requires human review.",
        "promotion_note": None,
        "quantity": None,
        "unit": None,
        "unit_price": None,
        "original_price": None,
        "discount_amount": None,
        "line_total": amt,
        "tax_rate": None,
        "tax_code": str(tax_code).lower() if tax_code else None,
        "category": "item",
        "source_line_ids": [str(x) for x in (source_line_ids or [])],
        "confidence": round(float(confidence), 3),
        "requires_review": True,
        "notes": "Validation-gated item row reconstructed from region re-OCR/table arbitration evidence.",
        "recovery_source": "vertical_price_stack_recovery",
        "recovery_evidence_source": evidence_source,
        "recovery_priority": priority,
        "recovery_amount_raw": amount_raw,
    }


def _merge_evidence_items(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate evidence rows by normalized product name.

    Lower priority wins.  Rows with amounts replace rows without amounts at the
    same priority/name.  This lets high-quality region re-OCR override a wrong
    LLM row, while table arbitration can fill rows that the region block marked
    as unmatched.
    """
    by_name: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for entry in entries:
        name = _norm(entry.get("product_description") or entry.get("description"))
        if not name:
            continue
        cur = by_name.get(name)
        if cur is None:
            by_name[name] = entry
            continue
        cur_amt = _amount(cur.get("line_total"))
        new_amt = _amount(entry.get("line_total"))
        cur_pri = int(
            cur.get("recovery_priority") if cur.get("recovery_priority") is not None else 99
        )
        new_pri = int(
            entry.get("recovery_priority") if entry.get("recovery_priority") is not None else 99
        )
        replace = False
        if cur_amt is None and new_amt is not None:
            replace = True
        elif new_amt is not None and cur_amt is not None:
            if new_pri < cur_pri:
                replace = True
            elif new_pri == cur_pri and abs(float(new_amt) - float(cur_amt)) > AMOUNT_TOL:
                # Same source quality but disagreement: keep the first, record conflict.
                conflicts.append(
                    {
                        "name": name,
                        "kept": cur_amt,
                        "discarded": new_amt,
                        "source": entry.get("recovery_evidence_source"),
                    }
                )
        elif new_pri < cur_pri:
            replace = True
        if replace:
            if (
                cur_amt is not None
                and new_amt is not None
                and abs(float(cur_amt) - float(new_amt)) > AMOUNT_TOL
            ):
                conflicts.append(
                    {
                        "name": name,
                        "kept": new_amt,
                        "discarded": cur_amt,
                        "source": entry.get("recovery_evidence_source"),
                    }
                )
            by_name[name] = entry
    items = [copy.deepcopy(v) for v in by_name.values() if _amount(v.get("line_total")) is not None]
    unmatched = [copy.deepcopy(v) for v in by_name.values() if _amount(v.get("line_total")) is None]
    return items, unmatched, conflicts


def _visual_best_block(visual_evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(visual_evidence, dict):
        return None
    block = visual_evidence.get("best_preferred_item_block")
    if isinstance(block, dict):
        return block
    region = (
        visual_evidence.get("region_reocr")
        if isinstance(visual_evidence.get("region_reocr"), dict)
        else {}
    )
    block = region.get("best_preferred_item_block") if isinstance(region, dict) else None
    return block if isinstance(block, dict) else None


def _table_arbitration_from_sources(
    visual_evidence: dict[str, Any] | None, table_arbitration: dict[str, Any] | None
) -> dict[str, Any] | None:
    if isinstance(table_arbitration, dict) and table_arbitration:
        return table_arbitration
    if isinstance(visual_evidence, dict) and isinstance(
        visual_evidence.get("table_arbitration"), dict
    ):
        return visual_evidence.get("table_arbitration")
    return None


def _fused_region_table_candidate(
    *,
    receipt: dict[str, Any],
    target_total: float,
    visual_evidence: dict[str, Any] | None,
    table_arbitration: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a full-table candidate from already available high-quality evidence.

    This is deliberately evaluated before crop OCR.  On hard supermarket receipts,
    the region re-OCR block often contains correct product/amount rows, table
    arbitration fills additional rows from original OCR, and the remaining exact
    residual can be assigned to a single unmatched product row.  The candidate is
    still applied only when it balances to the printed total.
    """
    entries: list[dict[str, Any]] = []
    order_sequences: list[list[dict[str, Any]]] = []
    diagnostics: dict[str, Any] = {
        "source": "fused_region_reocr_table_arbitration",
        "region_rows": 0,
        "region_unmatched_rows": 0,
        "table_rows": 0,
        "existing_rows_used": 0,
        "residual_assignment": None,
    }

    block = _visual_best_block(visual_evidence)
    if isinstance(block, dict):
        region_records = sort_records_by_source_position(
            [
                row
                for row in (block.get("rows") or []) + (block.get("unmatched_product_rows") or [])
                if isinstance(row, dict)
            ]
        )
        if region_records:
            order_sequences.append(region_records)
        for row in block.get("rows") or []:
            if not isinstance(row, dict):
                continue
            item = _evidence_item(
                description=row.get("description_candidate")
                or row.get("description")
                or row.get("text"),
                amount_value=row.get("amount"),
                amount_raw=row.get("amount_raw"),
                source_line_ids=row.get("source_line_ids") or [row.get("row_id")],
                evidence_source="region_reocr_preferred_item_block",
                priority=0,
                confidence=float(row.get("layout_confidence") or 0.82),
            )
            if item:
                entries.append(item)
                diagnostics["region_rows"] += 1
        for row in block.get("unmatched_product_rows") or []:
            if not isinstance(row, dict):
                continue
            item = _evidence_item(
                description=row.get("description_candidate")
                or row.get("description")
                or row.get("text"),
                amount_value=None,
                amount_raw=None,
                source_line_ids=[row.get("row_id")],
                evidence_source="region_reocr_unmatched_product_row",
                priority=1,
                confidence=0.66,
            )
            if item:
                entries.append(item)
                diagnostics["region_unmatched_rows"] += 1

    table = _table_arbitration_from_sources(visual_evidence, table_arbitration)
    if isinstance(table, dict):
        table_records = sort_records_by_source_position(
            [
                cand
                for cand in (table.get("ocr_layout_item_candidates") or [])
                if isinstance(cand, dict)
            ]
        )
        if table_records:
            order_sequences.append(table_records)
        for cand in table.get("ocr_layout_item_candidates") or []:
            if not isinstance(cand, dict):
                continue
            desc = cand.get("description")
            amt = _amount(cand.get("line_total"))
            if amt is None or abs(amt) < 0.01 or abs(amt) > max(300.0, target_total * 1.5):
                continue
            # Avoid payment/date/tax footer false positives like AS-Zeit or Betrag.
            ev_text = str(cand.get("evidence_text") or desc or "")
            if NON_PRODUCT_RE.search(ev_text) or FOOTER_RE.search(ev_text):
                continue
            item = _evidence_item(
                description=desc,
                amount_value=amt,
                amount_raw=cand.get("raw_amount"),
                tax_code=cand.get("tax_code"),
                source_line_ids=cand.get("source_line_ids") or [cand.get("row_id")],
                evidence_source="ocr_layout_table_arbitration",
                priority=2,
                confidence=0.74 if cand.get("product_percent_not_tax") else 0.70,
            )
            if item:
                if cand.get("product_percent_not_tax"):
                    item["product_percent_not_tax"] = True
                    item["notes"] += (
                        " Percent text was explicitly classified as product text, not tax evidence."
                    )
                entries.append(item)
                diagnostics["table_rows"] += 1

    existing_sequence = [old for old in (receipt.get("items") or []) if isinstance(old, dict)]
    if existing_sequence:
        order_sequences.append(existing_sequence)

    # Existing receipt rows are weakest, but can fill rows that all OCR evidence
    # already accepted earlier.  Higher-priority evidence still overrides them.
    for old in receipt.get("items") or []:
        if not isinstance(old, dict):
            continue
        desc = (
            old.get("product_description") or old.get("description") or old.get("raw_description")
        )
        amt = _amount(old.get("line_total"))
        if amt is None:
            continue
        item = _evidence_item(
            description=desc,
            amount_value=amt,
            amount_raw=old.get("line_total"),
            tax_code=old.get("tax_code"),
            source_line_ids=old.get("source_line_ids") or [],
            evidence_source="existing_reconciled_item_low_priority",
            priority=5,
            confidence=float(old.get("confidence") or 0.55),
        )
        if item:
            entries.append(item)
            diagnostics["existing_rows_used"] += 1

    amount_items, unmatched, conflicts = _merge_evidence_items(entries)
    amount_items = sort_items_by_printed_order(amount_items, sequences=order_sequences)
    unmatched = sort_items_by_printed_order(unmatched, sequences=order_sequences)
    current_sum = _item_sum(amount_items)
    residual = round(target_total - current_sum, 2)
    diagnostics.update(
        {
            "entry_count": len(entries),
            "amount_item_count_before_residual": len(amount_items),
            "unmatched_count_before_residual": len(unmatched),
            "sum_before_residual": current_sum,
            "residual_before_assignment": residual,
            "conflicts": conflicts,
        }
    )

    if abs(residual) <= AMOUNT_TOL:
        diagnostics["status"] = "balanced_without_residual_assignment"
        return amount_items, diagnostics

    # Validation-based residual assignment: only when exactly one product-like row
    # remains without an amount and the residual is a plausible item price.
    if residual > AMOUNT_TOL and len(unmatched) == 1 and residual <= max(300.0, target_total):
        filled = copy.deepcopy(unmatched[0])
        filled["line_total"] = round(residual, 2)
        filled["unit_price"] = None
        filled["confidence"] = min(float(filled.get("confidence") or 0.66), 0.58)
        filled["recovery_evidence_source"] = (
            f"{filled.get('recovery_evidence_source')}+balanced_residual_assignment"
        )
        filled["recovery_amount_raw"] = f"residual_to_printed_total:{residual:.2f}"
        filled["notes"] = (
            str(filled.get("notes") or "")
            + " Amount assigned from exact residual to printed receipt total; requires review."
        )
        amount_items.append(filled)
        diagnostics["residual_assignment"] = {
            "applied": True,
            "description": filled.get("product_description"),
            "assigned_amount": round(residual, 2),
            "reason": "single_unmatched_product_row_and_exact_printed_total_residual",
        }
        residual = round(target_total - _item_sum(amount_items), 2)

    diagnostics.update(
        {
            "candidate_count": len(amount_items),
            "candidate_sum": _item_sum(amount_items),
            "candidate_diff": residual,
            "status": "matched" if abs(residual) <= AMOUNT_TOL else "not_balanced",
        }
    )
    amount_items = sort_items_by_printed_order(amount_items, sequences=order_sequences)
    return amount_items, diagnostics


def run_vertical_price_stack_recovery(
    *,
    receipt: dict[str, Any],
    validation_report: dict[str, Any],
    ocr_context: dict[str, Any],
    image_path: Path | None,
    result_dir: Path,
    run_id: str,
    lang: str = "german",
    device: str = "cpu",
    min_score: float = 0.20,
    tolerance: float = 0.03,
    visual_evidence: dict[str, Any] | None = None,
    table_arbitration: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not isinstance(receipt, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "receipt_not_object",
        }
    if image_path is None or not Path(image_path).exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "missing_source_image",
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
    if not issue_codes.intersection(
        {"ITEM_SUM_MISMATCH", "NO_ITEMS", "MISSING_TOTAL", "UNRESOLVED_AMOUNT_LINES"}
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "no_item_sum_trigger",
            "issue_codes": sorted(issue_codes),
        }

    products = _product_rows(ocr_context)
    bounds = _right_column_bounds(ocr_context, products)
    if not products or not bounds:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "no_product_rows_or_bounds",
            "product_count": len(products),
            "before_diff": before_diff,
        }

    _emit(
        progress_callback,
        "vertical_price_stack_recovery",
        "running",
        "Running full right-side price-stack crop/OCR for unbalanced receipt.",
        product_count=len(products),
        before_diff=before_diff,
    )
    crop_result = _crop_and_ocr_stack(
        image_path=Path(image_path),
        bounds=bounds,
        result_dir=result_dir,
        run_id=run_id,
        lang=lang,
        device=device,
        min_score=min_score,
    )
    full_image_amounts = _amount_tokens_from_ocr_context(ocr_context, bounds=bounds)

    # Evaluate every evidence set, plus a merged set, against the printed total.
    # The accepted recovery is the one that most improves reconciliation.  This
    # avoids choosing a visually/OCR-rich variant that still pairs badly.
    evidence_sets: list[dict[str, Any]] = []
    fused_items, fused_diagnostics = _fused_region_table_candidate(
        receipt=receipt,
        target_total=target_total,
        visual_evidence=visual_evidence,
        table_arbitration=table_arbitration,
    )
    evidence_sets.append(
        {
            "source": "fused_region_reocr_table_arbitration",
            "amounts": [],
            "candidate_items": fused_items,
            "fused_diagnostics": fused_diagnostics,
        }
    )
    for variant in crop_result.get("crop_variants") or []:
        evidence_sets.append(
            {
                "source": f"crop:{variant.get('variant')}",
                "amounts": variant.get("amounts") or [],
            }
        )
    evidence_sets.append(
        {"source": "crop:merged_variants", "amounts": crop_result.get("amounts") or []}
    )
    evidence_sets.append({"source": "full_image_right_column", "amounts": full_image_amounts})
    evidence_sets.append(
        {
            "source": "merged_crop_and_full_image",
            "amounts": _merge_amount_evidence(crop_result.get("amounts") or [], full_image_amounts),
        }
    )

    attempts: list[dict[str, Any]] = []
    best_attempt: dict[str, Any] | None = None
    applied_receipt: dict[str, Any] | None = None
    mode: str | None = None
    full_selection: dict[str, Any] = {"status": "not_attempted"}
    candidate_items: list[dict[str, Any]] = []
    pairing: dict[str, Any] = {}
    amounts: list[dict[str, Any]] = []

    def attempt_score(attempt: dict[str, Any]) -> tuple[float, int, int]:
        after_abs = abs(float(attempt.get("after_diff", before_diff) or before_diff))
        # Prefer applied/balanced, then more paired rows.
        applied_bonus = 0 if attempt.get("applied") else 1
        paired = int((attempt.get("pairing") or {}).get("paired_count") or 0)
        return (after_abs + applied_bonus * 1000.0, -paired, -int(attempt.get("amount_count") or 0))

    for evidence in evidence_sets:
        ev_amounts = evidence.get("amounts") or []
        if evidence.get("candidate_items") is not None:
            ev_items = copy.deepcopy(evidence.get("candidate_items") or [])
            ev_pairing = {
                "status": "prepaired_fused_evidence",
                "paired_count": len(ev_items),
                "fused_diagnostics": evidence.get("fused_diagnostics") or {},
            }
        else:
            ev_items, ev_pairing = _candidate_items_by_y(products, ev_amounts)
        ev_receipt, ev_selection = _full_stack_candidate(
            receipt, ev_items, target_total, before_diff
        )
        ev_mode = "full_stack"
        if ev_receipt is None:
            ev_receipt, ev_selection = _addition_candidate_mode(
                receipt, ev_items, target_total, before_diff
            )
            ev_mode = "additions"
        if ev_receipt is not None:
            ev_after_sum = _item_sum(
                ev_receipt.get("items") if isinstance(ev_receipt.get("items"), list) else []
            )
            ev_after_diff = round(target_total - ev_after_sum, 2)
        else:
            ev_after_sum = before_sum
            ev_after_diff = before_diff
        attempt = {
            "source": evidence.get("source"),
            "amount_count": len(ev_amounts),
            "amount_sum": round(sum(float(a.get("value") or 0.0) for a in ev_amounts), 2),
            "candidate_item_count": len(ev_items),
            "candidate_sum": _item_sum(ev_items),
            "candidate_diff": round(target_total - _item_sum(ev_items), 2) if ev_items else None,
            "pairing": ev_pairing,
            "fused_diagnostics": evidence.get("fused_diagnostics"),
            "selection": ev_selection,
            "mode": ev_mode if ev_receipt is not None else None,
            "applied": ev_receipt is not None,
            "after_sum": ev_after_sum,
            "after_diff": ev_after_diff,
        }
        attempts.append(attempt)
        if best_attempt is None or attempt_score(attempt) < attempt_score(best_attempt):
            best_attempt = attempt
            applied_receipt = ev_receipt
            mode = ev_mode if ev_receipt is not None else None
            full_selection = ev_selection
            candidate_items = ev_items
            pairing = ev_pairing
            amounts = ev_amounts

    if best_attempt is None:
        best_attempt = {
            "source": None,
            "selection": {"status": "no_evidence_sets"},
            "after_sum": before_sum,
            "after_diff": before_diff,
        }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "applied" if applied_receipt is not None else "no_improvement",
        "applied": applied_receipt is not None,
        "mode": mode if applied_receipt is not None else None,
        "selected_evidence_source": best_attempt.get("source"),
        "requires_human_review": applied_receipt is not None,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "before_sum": before_sum,
        "target_total": target_total,
        "before_diff": before_diff,
        "issue_codes": sorted(issue_codes),
        "product_rows": products,
        "price_stack_crop": {k: v for k, v in crop_result.items() if k != "crop_variants"},
        "crop_variants": crop_result.get("crop_variants") or [],
        "full_image_right_column_amounts": full_image_amounts,
        "evidence_attempts": attempts,
        "pairing": pairing,
        "candidate_item_count": len(candidate_items),
        "candidate_sum": _item_sum(candidate_items),
        "candidate_diff": round(target_total - _item_sum(candidate_items), 2)
        if candidate_items
        else None,
        "selection": full_selection,
        "guidance": [
            "Vertical price-stack recovery is validation-gated and skipped for already balanced receipts.",
            "It first evaluates fused region re-OCR/table arbitration evidence, then OCR crop variants and full-image right-column boxes.",
            "Partial improvements are candidate-only; the final receipt is mutated only when the reconstructed item table balances to the printed total.",
            "Applied rows are marked requires_review=true and should not enter analytics/RAG without approval.",
        ],
    }
    if applied_receipt is not None:
        after_sum = _item_sum(
            applied_receipt.get("items") if isinstance(applied_receipt.get("items"), list) else []
        )
        result["after_sum"] = after_sum
        result["after_diff"] = round(target_total - after_sum, 2)
        result["receipt"] = applied_receipt
        _emit(
            progress_callback,
            "vertical_price_stack_recovery",
            "done",
            "Vertical price-stack recovery improved item-total reconciliation.",
            after_diff=result.get("after_diff"),
            mode=mode,
            selected_evidence_source=result.get("selected_evidence_source"),
        )
    else:
        result["after_sum"] = before_sum
        result["after_diff"] = before_diff
        _emit(
            progress_callback,
            "vertical_price_stack_recovery",
            "warning",
            "Vertical price-stack recovery did not improve validation.",
            amount_count=len(amounts),
            candidate_item_count=len(candidate_items),
            reason=full_selection.get("status"),
            selected_evidence_source=result.get("selected_evidence_source"),
        )
    return result
