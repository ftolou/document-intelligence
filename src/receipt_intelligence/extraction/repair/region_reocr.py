#!/usr/bin/env python3
"""VLM-region-first crop OCR evidence.

PaddleOCR-VL is used as a layout/region detector, not as the text authority.
This module extracts VLM layout bboxes, maps them back to the original image,
crops the original high-resolution image, runs PaddleOCR on those crops, and
builds compact preferred item-block evidence for the LLM.

The module deliberately does not create the final receipt JSON. It only creates
source-grounded OCR evidence from region crops plus generic candidate groups.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

from receipt_intelligence.engines.ocr_engine import _run_paddle_ocr_v13_profile

AMOUNT_RE = re.compile(
    r"(?<!\d)([-+−]?\s*\d{1,5}(?:[.\s]\d{3})*(?:[,\.]\s*\d{2})|[-+−]?\s*\d{1,5}\s+\d{2})(?:\s*[-−])?(?!\d)"
)
DATE_RE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
ITEM_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")
QTY_RE = re.compile(
    r"(?:^|\b)\d+[,.]?\d*\s*(?:STK|STÜCK|STUECK|PCS?|KG|G|L|ML|M)?\s*(?:x|×|@|à|a)\s*\d+[,.]\d{1,3}|\b\d+[,.]?\d*\s*(?:KG|G|L|ML|STK|STÜCK|STUECK|PCS?)\b",
    re.I,
)
UNIT_WORD_RE = re.compile(
    r"\b(STK|STÜCK|STUECK|PCS?|KG|G|GRAMM|L|ML|M|EUR/KG|EUR/L|€/KG|€/L)\b", re.I
)
TOTAL_RE = re.compile(
    r"\b(SUMME|BONSUMME|GESAMT|TOTAL|ZU\s+(?:ZAHLEN|BEZAHLEN)|ENDS?SUMME|ZWISCHENSUMME)\b", re.I
)
PAYMENT_RE = re.compile(
    r"\b(BAR|CASH|GEGEBEN|GEG\.\s*EC|EC[- ]?CASH|GIROCARD|KARTE|CARD|VISA|MASTERCARD|ZAHLUNG|LASTSCHRIFT)\b",
    re.I,
)
CHANGE_RE = re.compile(r"\b(RÜCKGELD|RUECKGELD|ZURÜCK|ZURUECK|CHANGE|WECHSELGELD)\b", re.I)
TAX_RE = re.compile(r"\b(MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|GROSS|RATE|SATZ)\b", re.I)
FOOTER_RE = re.compile(
    r"\b(BON|BELEG|TERMINAL|TRACE|KASSE|FILIALE|UID|UST-ID|DATUM|UHRZEIT|TEL|FAX|WWW|KUNDENBELEG|ÖFFNUNGSZEIT|OEFFNUNGSZEIT)\b",
    re.I,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _parse_amount_token(raw: str) -> float | None:
    s = str(raw or "").strip().replace("−", "-")
    negative = (
        s.startswith("-") or s.endswith("-") or re.search(r"[,\.]\s*\d{2}\s*[-−]", s) is not None
    )
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
    return -abs(v) if negative else v


def _amounts(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    if DATE_RE.search(text) and "," not in text:
        return []
    out: list[dict[str, Any]] = []
    for m in AMOUNT_RE.finditer(text):
        val = _parse_amount_token(m.group(0))
        if val is not None:
            out.append({"raw": m.group(0).strip(), "value": val})
    return out


def _is_damaged_amount_token(text: str) -> bool:
    """True for OCR fragments that look like a price but are not parseable.

    This is intentionally conservative. It only marks short numeric/decimal
    fragments, often with a tax marker letter, as damaged amount evidence.
    Product names, dates, quantities and section labels must not match here.
    """
    t = _norm_text(text or "")
    if not t or _amounts(t):
        return False
    if DATE_RE.search(t) or TIME_ONLY_RE.fullmatch(t):
        return False
    if TOTAL_RE.search(t) or PAYMENT_RE.search(t) or CHANGE_RE.search(t) or TAX_RE.search(t):
        return False
    if ITEM_WORD_RE.search(t) and not re.fullmatch(r"\s*\d+[,.]?\s*[A-Z]\s*", t, re.I):
        return False
    digits = sum(ch.isdigit() for ch in t)
    if digits == 0 or digits > 4:
        return False
    return bool(DAMAGED_AMOUNT_TOKEN_RE.fullmatch(t))


def _damaged_amount_candidate(text: str) -> dict[str, Any] | None:
    if not _is_damaged_amount_token(text):
        return None
    return {"raw": _norm_text(text), "value": None, "status": "damaged_amount_token"}


def _strip_amounts(text: str) -> str:
    return re.sub(AMOUNT_RE, " ", text or "").replace("  ", " ").strip(" |:-")


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _walk_json(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            if k in {"image", "base64", "embedding"}:
                continue
            yield from _walk_json(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_json(v, f"{path}[{i}]")


def _coerce_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        vals = [float(value[i]) for i in range(4)]
    except Exception:
        return None
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        return None
    return vals


def _extract_layout_boxes(vlm_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract layout bboxes from PaddleOCR-VL raw result JSON files.

    The CLI JSON often contains raw_result.files[].content.layout_det_res.boxes with
    coordinates on the VLM-resized image. Older code only extracted table text;
    V14.13.3 also preserves the layout region bboxes.
    """
    boxes: list[dict[str, Any]] = []
    raw = vlm_result.get("raw_result")
    for path, obj in _walk_json(raw):
        if not isinstance(obj, dict):
            continue
        bbox = _coerce_bbox(obj.get("coordinate") or obj.get("bbox") or obj.get("block_bbox"))
        if not bbox:
            continue
        label = (
            str(obj.get("label") or obj.get("type") or obj.get("category") or "").strip().lower()
        )
        score = obj.get("score")
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None
        boxes.append(
            {
                "id": f"vlm_layout_box_{len(boxes):03d}",
                "source_path": path,
                "label": label or None,
                "coordinate": bbox,
                "score": score,
                "order": obj.get("order"),
            }
        )
    return boxes


def _image_prepare(vlm_result: dict[str, Any], source_image_path: Path) -> dict[str, Any]:
    prep = (
        vlm_result.get("image_prepare") if isinstance(vlm_result.get("image_prepare"), dict) else {}
    )
    with Image.open(source_image_path) as img:
        ow, oh = img.size
    original_width = int(prep.get("original_width") or ow)
    original_height = int(prep.get("original_height") or oh)
    prepared_width = int(prep.get("prepared_width") or prep.get("width") or original_width)
    prepared_height = int(prep.get("prepared_height") or prep.get("height") or original_height)
    scale = prep.get("scale")
    try:
        scale = float(scale)
    except Exception:
        scale = prepared_width / max(original_width, 1)
    if scale <= 0:
        scale = prepared_width / max(original_width, 1)
    return {
        "original_width": original_width,
        "original_height": original_height,
        "prepared_width": prepared_width,
        "prepared_height": prepared_height,
        "scale": scale,
    }


def _map_bbox_to_original(bbox: list[float], prep: dict[str, Any], pad: int = 18) -> list[int]:
    scale = float(prep.get("scale") or 1.0)
    ow = int(prep.get("original_width") or 1)
    oh = int(prep.get("original_height") or 1)
    x1, y1, x2, y2 = bbox
    if scale and scale != 1.0:
        x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
    out = [
        max(0, int(math.floor(x1 - pad))),
        max(0, int(math.floor(y1 - pad))),
        min(ow, int(math.ceil(x2 + pad))),
        min(oh, int(math.ceil(y2 + pad))),
    ]
    if out[2] <= out[0] or out[3] <= out[1]:
        return [0, 0, ow, oh]
    return out


def _select_region_boxes(
    layout_boxes: list[dict[str, Any]], prep: dict[str, Any]
) -> list[dict[str, Any]]:
    """Choose crop regions. Prefer table bboxes, but keep text-heavy broad bboxes as fallback."""
    pw = float(prep.get("prepared_width") or 1)
    ph = float(prep.get("prepared_height") or 1)
    candidates: list[dict[str, Any]] = []
    for b in layout_boxes:
        bbox = b.get("coordinate") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        area = max(0.0, w * h)
        label = str(b.get("label") or "").lower()
        # Tables are primary. Also accept tall text regions because some VLM versions
        # label receipt item regions as text/doc_title rather than table.
        is_table = "table" in label
        is_broad_text = (
            label in {"text", "doc_title", "paragraph"} and h > ph * 0.08 and w > pw * 0.45
        )
        if not (is_table or is_broad_text):
            continue
        if area < pw * ph * 0.01:
            continue
        kind = "item_table_candidate" if is_table else "text_region_candidate"
        candidates.append(
            {
                **b,
                "kind": kind,
                "area": area,
                "height_ratio": h / max(ph, 1.0),
                "width_ratio": w / max(pw, 1.0),
            }
        )
    # Prefer larger tables first; keep at most 3 to avoid excessive OCR.
    candidates.sort(
        key=lambda r: (1 if r.get("kind") == "item_table_candidate" else 0, r.get("area") or 0),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for c in candidates:
        # Skip almost-identical duplicates inside already kept broader boxes.
        cx1, cy1, cx2, cy2 = [float(v) for v in c.get("coordinate")]
        duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2 = [float(v) for v in k.get("coordinate")]
            inter_w = max(0.0, min(cx2, kx2) - max(cx1, kx1))
            inter_h = max(0.0, min(cy2, ky2) - max(cy1, ky1))
            inter = inter_w * inter_h
            small = min((cx2 - cx1) * (cy2 - cy1), (kx2 - kx1) * (ky2 - ky1))
            if small and inter / small > 0.75:
                duplicate = True
                break
        if not duplicate:
            kept.append(c)
        if len(kept) >= 3:
            break
    return kept


def _crop_image(source_image_path: Path, bbox: list[int], crop_path: Path) -> dict[str, Any]:
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image_path) as img:
        img = img.convert("RGB")
        crop = img.crop(tuple(bbox))
        # Mild contrast/scale helps small receipt text without changing geometry.
        if max(crop.size) < 1800:
            scale = min(2.0, 1800 / max(max(crop.size), 1))
            if scale > 1.05:
                crop = crop.resize(
                    (int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS
                )
            else:
                scale = 1.0
        else:
            scale = 1.0
        crop = ImageEnhance.Contrast(crop).enhance(1.15)
        crop.save(crop_path, quality=95)
    return {
        "path": str(crop_path),
        "bbox_original": bbox,
        "crop_scale": scale,
        "width": crop.width,
        "height": crop.height,
    }


# Common non-product tokens that appear in receipt regions but should not become item descriptions.
NON_PRODUCT_TOKEN_RE = re.compile(
    r"^(?:EUR|EURO|A|B|C|D|E|V|RATE|SATZ|DATUM|UHR|UHRZEIT|NR|NO|NUMMER|BON|BELEG|KASSE|TRACE|TERMINAL|KUNDENBELEG|GIROCARD|SUMME)$",
    re.I,
)
CURRENCY_UNIT_ONLY_RE = re.compile(
    r"^[\s|:/.-]*(?:EUR|EURO|€|A|B|C|D|E|V|KG|G|L|ML|STK|PCS|EUR/KG|EUR/L|€/KG|€/L)+[\s|:/.-]*$",
    re.I,
)
UNIT_PRICE_ONLY_RE = re.compile(
    r"^\s*[-+]?\d+[,.]\d{1,3}\s*(?:EUR|EURO|€)?\s*/\s*(?:KG|G|L|ML|M|STK|PCS?)\s*$", re.I
)
TIME_ONLY_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:UHR)?\s*$", re.I)
# OCR may damage right-column amounts without producing a valid decimal amount.
# Examples observed in REWE crop OCR: "79" for 3,79 and "0,6A" for 0,69 A.
# Such tokens must be treated as an owned but unresolved amount for that row,
# not ignored so the product steals the next row's clean amount.
DAMAGED_AMOUNT_TOKEN_RE = re.compile(
    r"^\s*[-+−]?(?:\d{1,2}|\d+[,.]\s*[0-9A-Z]?)\s*[A-Z]?\s*$", re.I
)
FINAL_PRICE_LABEL_RE = re.compile(
    r"\b(IHR\s+PREIS|DEIN\s+PREIS|AKTIONSPREIS|SALE\s*PRICE|ENDPREIS)\b", re.I
)
DISCOUNT_LABEL_RE = re.compile(r"\b(RABATT|DISCOUNT|NACHLASS|REDUZIERT|GUTSCHRIFT)\b", re.I)
REFERENCE_REASON_RE = re.compile(r"\b(GRUND|REASON|CODE|ART\.?NR|EAN|FAR\d{4,})\b", re.I)


def _meaningful_words(text: str) -> list[str]:
    words = ITEM_WORD_RE.findall(text or "")
    return [w for w in words if not NON_PRODUCT_TOKEN_RE.fullmatch(w)]


def _is_currency_or_unit_only(text: str) -> bool:
    t = _strip_amounts(text or "")
    t = re.sub(r"[\s|:/.-]+", " ", t).strip()
    if not t:
        return True
    if CURRENCY_UNIT_ONLY_RE.fullmatch(t):
        return True
    words = ITEM_WORD_RE.findall(t)
    return bool(words) and all(NON_PRODUCT_TOKEN_RE.fullmatch(w) for w in words)


def _is_unit_price_only(text: str) -> bool:
    t = _norm_text(text or "")
    if UNIT_PRICE_ONLY_RE.fullmatch(t):
        return True
    # OCR often returns "1,99 EUR/kg" as one token/line. This is support for
    # a quantity row, not a standalone product line.
    stripped = _strip_amounts(t)
    return bool(UNIT_WORD_RE.search(stripped)) and not _meaningful_words(
        stripped.replace("EUR", " ").replace("EURO", " ")
    )


def _is_quantity_note(text: str) -> bool:
    t = text or ""
    if TOTAL_RE.search(t) or PAYMENT_RE.search(t) or CHANGE_RE.search(t) or TAX_RE.search(t):
        return False
    if _is_unit_price_only(t):
        return True
    # A/B/C tax markers after a right-column amount are not multipliers.
    if re.fullmatch(r"\s*[-+]?\d+[,.]\d{1,3}\s*[A-Z]\s*", t, re.I):
        return False
    words = ITEM_WORD_RE.findall(t)
    non_unit_words = [
        w for w in words if not UNIT_WORD_RE.fullmatch(w) and not NON_PRODUCT_TOKEN_RE.fullmatch(w)
    ]
    # Quantity rows can be complete ("0,520 kg x 2,49") or split across OCR
    # lines ("0,520 kg x" plus a separate unit-price amount). Both must be
    # support rows, never standalone items.
    if re.search(
        r"\b\d+[,.]?\d*\s*(?:KG|G|L|ML|STK|STÜCK|STUECK|PCS?)\s*(?:x|×|@|à|a)?\s*$", t, re.I
    ):
        return len(non_unit_words) <= 1
    if re.search(r"^\s*\d+[,.]?\d*\s*(?:x|×|@|à)\s*$", t, re.I):
        return True
    return bool(QTY_RE.search(t)) and len(non_unit_words) <= 1


def _is_non_item_section(text: str) -> bool:
    return bool(
        TOTAL_RE.search(text or "")
        or PAYMENT_RE.search(text or "")
        or CHANGE_RE.search(text or "")
        or TAX_RE.search(text or "")
        or FOOTER_RE.search(text or "")
    )


def _looks_product_like(text: str) -> bool:
    t = text or ""
    if (
        not t
        or _is_non_item_section(t)
        or _is_quantity_note(t)
        or _is_currency_or_unit_only(t)
        or TIME_ONLY_RE.fullmatch(t)
    ):
        return False
    words = _meaningful_words(t)
    return len(words) >= 1


def _line_records_from_ocr(
    ocr: dict[str, Any], *, crop_bbox: list[int], crop_scale: float
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, w in enumerate(ocr.get("words") or []):
        text = _norm_text(w.get("text"))
        if not text:
            continue
        # Convert crop coordinates back to original image coordinates.
        sx = sy = float(crop_scale or 1.0)
        ox1, oy1, _, _ = crop_bbox
        xmin = ox1 + float(w.get("xmin") or 0) / sx
        ymin = oy1 + float(w.get("ymin") or 0) / sy
        xmax = ox1 + float(w.get("xmax") or 0) / sx
        ymax = oy1 + float(w.get("ymax") or 0) / sy
        amts = _amounts(text)
        damaged_amt = _damaged_amount_candidate(text)
        if _is_quantity_note(text):
            role = "quantity_or_unit_price_note"
        elif _is_non_item_section(text):
            role = "section_boundary"
        elif _looks_product_like(_strip_amounts(text)):
            role = "product_or_item_text"
        elif amts:
            role = "amount_only"
        elif damaged_amt:
            role = "damaged_amount_candidate"
        else:
            role = "unclassified"
        records.append(
            {
                "id": f"region_line_{i:03d}",
                "text": text,
                "confidence": float(w.get("confidence") or 0.0),
                "xmin": round(xmin, 1),
                "ymin": round(ymin, 1),
                "xmax": round(xmax, 1),
                "ymax": round(ymax, 1),
                "x_center": round((xmin + xmax) / 2, 1),
                "y_center": round((ymin + ymax) / 2, 1),
                "amounts": amts,
                "damaged_amount_candidate": damaged_amt,
                "role_hint": role,
            }
        )
    records.sort(key=lambda r: (r["ymin"], r["xmin"]))
    for i, r in enumerate(records):
        r["id"] = f"region_line_{i:03d}"
    return records


def _median_line_height(records: list[dict[str, Any]]) -> float:
    heights = [max(5.0, float(r.get("ymax", 0)) - float(r.get("ymin", 0))) for r in records]
    return sorted(heights)[len(heights) // 2] if heights else 18.0


def _find_item_start_y(records: list[dict[str, Any]]) -> float | None:
    """Find first likely product/quantity row; ignore header/footer boundaries above it.

    This fixes the REWE failure where a UID line above the item region was
    treated as an item-section stop boundary and emptied the item block.
    """
    for r in records:
        txt = r.get("text") or ""
        if _looks_product_like(_strip_amounts(txt)) or _is_quantity_note(txt):
            return float(r.get("ymin") or 0.0)
    # Fallback: first right-column amount if text was fragmented heavily.
    for r in records:
        if r.get("amounts"):
            return float(r.get("ymin") or 0.0)
    return None


def _find_item_stop_y(records: list[dict[str, Any]], start_y: float | None) -> float | None:
    if start_y is None:
        return None
    median_h = _median_line_height(records)
    # Only section boundaries after the item region starts may stop item parsing.
    for r in records:
        y = float(r.get("ymin") or 0.0)
        if y < start_y + median_h * 0.50:
            continue
        txt = r.get("text") or ""
        if (
            TOTAL_RE.search(txt)
            or PAYMENT_RE.search(txt)
            or CHANGE_RE.search(txt)
            or TAX_RE.search(txt)
        ):
            return y
    return None


def _nearest_amount_for_row(
    r: dict[str, Any],
    amount_only: list[dict[str, Any]],
    used: set[str],
    y_tol: float,
    prefer_right_of: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    ry = float(r.get("y_center") or 0)
    rx = float(r.get("x_center") or 0)
    candidates = []
    for a in amount_only:
        aid = str(a.get("id"))
        if aid in used:
            continue
        dy = abs(float(a.get("y_center") or 0) - ry)
        if dy > y_tol:
            continue
        ax = float(a.get("x_center") or 0)
        # Prefer price-column amounts to the right of text, but still allow if
        # OCR coordinate ordering is odd.
        right_bonus = 0 if (not prefer_right_of or ax >= rx) else 20
        candidates.append((dy + right_bonus, -ax, a))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (x[0], x[1]))
    a = candidates[0][2]
    selected = (a.get("amounts") or [None])[-1]
    return selected, str(a.get("id"))


def _find_printed_total(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    median_h = _median_line_height(records)
    amount_rows = [r for r in records if r.get("amounts")]
    for r in records:
        txt = r.get("text") or ""
        if not TOTAL_RE.search(txt):
            continue
        amts = r.get("amounts") or []
        if amts:
            return {
                "row_id": r.get("id"),
                "text": txt,
                "amount": amts[-1].get("value"),
                "raw": amts[-1].get("raw"),
            }
        # Many receipts render "SUMME" left, "EUR" center and the amount on a
        # separate right-column OCR line. Link by baseline proximity.
        ry = float(r.get("y_center") or 0)
        candidates = []
        for a in amount_rows:
            dy = abs(float(a.get("y_center") or 0) - ry)
            if dy <= max(18.0, median_h * 1.25):
                candidates.append((dy, -float(a.get("x_center") or 0), a))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            a = candidates[0][2]
            amts = a.get("amounts") or []
            if amts:
                return {
                    "row_id": r.get("id"),
                    "amount_row_id": a.get("id"),
                    "text": f"{txt} | {a.get('text')}",
                    "amount": amts[-1].get("value"),
                    "raw": amts[-1].get("raw"),
                }
    return None


def _tokens_for_match(text: str) -> set[str]:
    t = _strip_amounts(text or "").upper()
    t = re.sub(r"[^A-ZÄÖÜẞ0-9]+", " ", t)
    tokens = set()
    for tok in t.split():
        if len(tok) < 3:
            continue
        if NON_PRODUCT_TOKEN_RE.fullmatch(tok) or UNIT_WORD_RE.fullmatch(tok):
            continue
        tokens.add(tok)
    return tokens


def _extract_vlm_item_amount_rows(visual_evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(visual_evidence, dict):
        return []
    rows = []
    for table in visual_evidence.get("structured_tables") or []:
        for row in table.get("rows") or []:
            hints = set(row.get("role_hints") or [])
            if "possible_item_charge_or_credit" not in hints:
                continue
            amts = row.get("amounts") or []
            if not amts:
                continue
            cells = row.get("cells") or []
            text = " ".join(str(c) for c in cells)
            desc = _strip_amounts(text)
            if not _looks_product_like(desc):
                continue
            tokens = _tokens_for_match(desc)
            if not tokens:
                continue
            rows.append(
                {
                    "vlm_row_id": row.get("id"),
                    "text": text,
                    "description_candidate": desc,
                    "amount": amts[-1].get("value"),
                    "amount_raw": amts[-1].get("raw"),
                    "tokens": tokens,
                }
            )
    return rows


def _match_vlm_amount(
    desc: str, vlm_rows: list[dict[str, Any]], used_vlm: set[str]
) -> dict[str, Any] | None:
    dtoks = _tokens_for_match(desc)
    if not dtoks:
        return None
    scored = []
    for vr in vlm_rows:
        vid = str(vr.get("vlm_row_id"))
        if vid in used_vlm:
            continue
        vtoks = vr.get("tokens") or set()
        overlap = len(dtoks & vtoks)
        if overlap <= 0:
            continue
        # Require meaningful overlap for longer product names. One strong token
        # like EIER/ELMEX/LUXUS/WATTEPADS is enough; avoid accidental generic matches.
        ratio = overlap / max(1, min(len(dtoks), len(vtoks)))
        if overlap >= 1 and ratio >= 0.45:
            scored.append((overlap, ratio, vr))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def _product_y_bounds(
    product_lines: list[dict[str, Any]], boundary_y: float | None = None
) -> dict[str, tuple[float, float]]:
    ordered = sorted(product_lines, key=lambda r: float(r.get("y_center") or 0.0))
    bounds: dict[str, tuple[float, float]] = {}
    for i, r in enumerate(ordered):
        y = float(r.get("y_center") or 0.0)
        if i == 0:
            low = y - max(40.0, (float(r.get("ymax") or y) - float(r.get("ymin") or y)) * 1.2)
        else:
            py = float(ordered[i - 1].get("y_center") or y)
            low = (py + y) / 2.0
        if i == len(ordered) - 1:
            high = y + max(35.0, (float(r.get("ymax") or y) - float(r.get("ymin") or y)) * 1.0)
        else:
            ny = float(ordered[i + 1].get("y_center") or y)
            high = (y + ny) / 2.0
        if boundary_y is not None:
            high = min(high, float(boundary_y) - 1.0)
        bounds[str(r.get("id"))] = (low, high)
    return bounds


def _is_adjustment_or_reference_label(text: str) -> bool:
    t = text or ""
    return bool(
        FINAL_PRICE_LABEL_RE.search(t)
        or DISCOUNT_LABEL_RE.search(t)
        or REFERENCE_REASON_RE.search(t)
    )


def _damaged_right_token_for_same_y_band(
    r: dict[str, Any], damaged_amounts: list[dict[str, Any]], y_tol: float
) -> dict[str, Any] | None:
    """Return a damaged right-column amount token that belongs to this product row.

    If this returns a token, the row must not borrow a clean amount from the
    next row. The damaged token is owned by this row and should be repaired from
    VLM/full OCR/targeted crop evidence or left unresolved.
    """
    ry = float(r.get("y_center") or 0.0)
    rx = float(r.get("x_center") or 0.0)
    candidates = []
    for a in damaged_amounts:
        ay = float(a.get("y_center") or 0.0)
        ax = float(a.get("x_center") or 0.0)
        if ax <= rx:
            continue
        dy = abs(ay - ry)
        if dy <= y_tol:
            candidates.append((dy, -ax, a))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def _right_amount_for_same_y_band(
    r: dict[str, Any], amount_only: list[dict[str, Any]], used: set[str], y_tol: float
) -> tuple[dict[str, Any] | None, str | None]:
    """Return an explicit right-column line total on the same row.

    This fixes rows such as "Pfand 0,25 EXM" where 0,25 is an inline unit
    amount, while the true line total 1,50 is printed in the right price column.
    """
    ry = float(r.get("y_center") or 0.0)
    rx = float(r.get("x_center") or 0.0)
    candidates = []
    for a in amount_only:
        aid = str(a.get("id"))
        if aid in used:
            continue
        ay = float(a.get("y_center") or 0.0)
        ax = float(a.get("x_center") or 0.0)
        if abs(ay - ry) > y_tol:
            continue
        if ax <= rx:
            continue
        selected = (a.get("amounts") or [None])[-1]
        if selected is None or selected.get("value") is None:
            continue
        candidates.append((-ax, abs(ay - ry), a, selected))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][3], str(candidates[0][2].get("id"))


def _amount_for_product_range(
    r: dict[str, Any],
    amount_only: list[dict[str, Any]],
    used: set[str],
    bounds: dict[str, tuple[float, float]],
) -> tuple[dict[str, Any] | None, str | None]:
    rid = str(r.get("id"))
    low, high = bounds.get(rid, (-(10**9), 10**9))
    rx = float(r.get("x_center") or 0.0)
    candidates = []
    for a in amount_only:
        aid = str(a.get("id"))
        if aid in used:
            continue
        ay = float(a.get("y_center") or 0.0)
        if not (low <= ay < high):
            continue
        ax = float(a.get("x_center") or 0.0)
        if ax < rx:  # ignore unit-price amounts in the middle/left columns
            continue
        selected = (a.get("amounts") or [None])[-1]
        if selected is None or selected.get("value") is None:
            continue
        # Prefer the rightmost amount in the product's y-band.
        candidates.append((-ax, abs(ay - float(r.get("y_center") or 0.0)), a, selected))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        a = candidates[0][2]
        return candidates[0][3], str(a.get("id"))
    return None, None


def _apply_final_price_overrides(
    rows: list[dict[str, Any]],
    product_lines: list[dict[str, Any]],
    final_price_lines: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply conservative original-price -> final-price grouping.

    This targets Modepark-like receipts where the item has a list/reference price
    on the right column followed by an explicit "Ihr Preis" row. It does not
    create new products; it only replaces the already detected product row amount
    with the explicit final customer price when that final-price row lies before
    the next product row.
    """
    if not rows or not final_price_lines:
        return rows, []

    rows_by_id = {str(r.get("row_id")): r for r in rows}
    ordered_products = sorted(product_lines, key=lambda rr: float(rr.get("y_center") or 0.0))
    groups: list[dict[str, Any]] = []
    used_product_ids: set[str] = set()

    for fp in sorted(final_price_lines, key=lambda rr: float(rr.get("y_center") or 0.0)):
        fp_amounts = fp.get("amounts") or []
        final_val = fp_amounts[-1].get("value") if fp_amounts else None
        if final_val is None:
            continue
        fy = float(fp.get("y_center") or 0.0)
        best_prod = None
        for idx, prod in enumerate(ordered_products):
            pid = str(prod.get("id"))
            if pid in used_product_ids or pid not in rows_by_id:
                continue
            py = float(prod.get("y_center") or 0.0)
            if py >= fy:
                continue
            next_y = (
                float(ordered_products[idx + 1].get("y_center") or 10**9)
                if idx + 1 < len(ordered_products)
                else 10**9
            )
            # The final-price label belongs to the previous product if it appears
            # before the next real product row. Reference/barcode rows are not in
            # rows_by_id, so they do not steal the group.
            if fy < next_y:
                best_prod = prod
        if best_prod is None:
            continue
        pid = str(best_prod.get("id"))
        row = rows_by_id.get(pid)
        if not row:
            continue
        orig = (
            _parse_amount_token(row.get("amount"))
            if not isinstance(row.get("amount"), (int, float))
            else float(row.get("amount"))
        )
        if orig is None:
            continue
        # Final price should usually be <= original/list price. Allow equality
        # and small OCR tolerances, but do not replace with larger values.
        if final_val > orig + 0.05:
            continue
        old_amount = row.get("amount")
        row["amount"] = round(float(final_val), 2)
        row["amount_raw"] = fp_amounts[-1].get("raw")
        row["evidence_source"] = "crop_ocr_final_price_group"
        src = list(row.get("source_line_ids") or [row.get("row_id")])
        if fp.get("id") not in src:
            src.append(fp.get("id"))
        row["source_line_ids"] = src
        row["original_or_reference_price"] = old_amount
        row["final_price_row_id"] = fp.get("id")
        groups.append(
            {
                "candidate_id": f"region_final_price_{len(groups):03d}",
                "pattern": "original_price_followed_by_explicit_final_price",
                "product_row_id": row.get("row_id"),
                "product_row_text": row.get("text"),
                "product_description_candidate": row.get("description_candidate"),
                "original_or_reference_price": old_amount,
                "final_price_row_id": fp.get("id"),
                "final_price_row_text": fp.get("text"),
                "final_sale_price_candidate": round(float(final_val), 2),
                "relationship_ok": True,
                "source": "region_crop_reocr",
            }
        )
        used_product_ids.add(pid)
    return rows, groups


def _build_preferred_item_block(
    records: list[dict[str, Any]], visual_evidence: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build preferred item candidates from crop OCR rows.

    This is candidate evidence, not final parsing. V14.13.3 keeps the item-zone
    bug by ignoring header/footer lines above the first likely item and only
    stopping at total/payment/tax boundaries after the item block starts.
    """
    start_y = _find_item_start_y(records)
    boundary_y = _find_item_stop_y(records, start_y)
    if start_y is None:
        item_zone = []
    else:
        item_zone = [
            r
            for r in records
            if float(r.get("ymin") or 0) >= start_y - _median_line_height(records) * 0.75
            and (boundary_y is None or float(r.get("ymin") or 0) < boundary_y)
        ]

    product_lines = [
        r
        for r in item_zone
        if _looks_product_like(_strip_amounts(r.get("text") or ""))
        and not _is_quantity_note(r.get("text") or "")
        and not _is_adjustment_or_reference_label(r.get("text") or "")
    ]
    amount_only = [
        r
        for r in item_zone
        if (
            r.get("amounts")
            and not _looks_product_like(_strip_amounts(r.get("text") or ""))
            and not _is_quantity_note(r.get("text") or "")
        )
    ]
    damaged_amounts = [
        r
        for r in item_zone
        if r.get("role_hint") == "damaged_amount_candidate"
        or r.get("damaged_amount_candidate")
        or _is_damaged_amount_token(r.get("text") or "")
    ]
    quantity_lines = [r for r in item_zone if _is_quantity_note(r.get("text") or "")]
    final_price_lines = [
        r
        for r in item_zone
        if FINAL_PRICE_LABEL_RE.search(r.get("text") or "") and r.get("amounts")
    ]

    rows: list[dict[str, Any]] = []
    unmatched_product_rows: list[dict[str, Any]] = []
    used_amount_line_ids: set[str] = set()
    used_vlm_ids: set[str] = set()
    median_h = _median_line_height(records)
    vlm_amount_rows = _extract_vlm_item_amount_rows(visual_evidence)
    product_bounds = _product_y_bounds(product_lines, boundary_y=boundary_y)
    ordered_products_for_links = sorted(
        product_lines, key=lambda rr: float(rr.get("y_center") or 0.0)
    )
    next_product_y = {
        str(rr.get("id")): (
            float(ordered_products_for_links[i + 1].get("y_center") or 0.0)
            if i + 1 < len(ordered_products_for_links)
            else None
        )
        for i, rr in enumerate(ordered_products_for_links)
    }

    for r in product_lines:
        text = r.get("text") or ""
        desc = _strip_amounts(text)
        # Avoid carrying quantity notation inside item description when same line
        # includes both product and quantity note.
        desc = re.sub(
            r"\s*\d+[,.]?\d*\s*(?:KG|G|L|ML|STK|STÜCK|STUECK|PCS?)\s*(?:x|×|@|à|a)\s*\d+[,.]\d+.*$",
            "",
            desc,
            flags=re.I,
        ).strip(" |:-")
        if not desc or not _looks_product_like(desc):
            continue

        amts = r.get("amounts") or []
        selected = None
        amount_source = r.get("id")
        evidence_source = "crop_ocr_same_line"

        # Highest priority guard: if the same product row has a damaged right-
        # column amount token, the row owns that damaged token. Do not let it
        # borrow the next row's clean amount. Repair via VLM/full OCR evidence or
        # leave unresolved. This fixes REWE-like shifts after tokens such as
        # "79" (3,79) and "0,6A" (0,69 A).
        damaged_same_row = _damaged_right_token_for_same_y_band(
            r, damaged_amounts, y_tol=min(34.0, max(18.0, median_h * 0.70))
        )

        if damaged_same_row is None:
            # Highest priority: an explicit right-column amount on the same y-band.
            # This prevents inline unit prices/deposit values from being mistaken for
            # the line total, e.g. "Pfand 0,25 EXM | 1,50".
            selected, amount_source = _right_amount_for_same_y_band(
                r, amount_only, used_amount_line_ids, y_tol=max(16.0, median_h * 0.85)
            )
            if selected is not None and amount_source is not None:
                used_amount_line_ids.add(str(amount_source))
                evidence_source = "crop_ocr_right_amount_same_y_band"
            elif amts and not _is_unit_price_only(text):
                selected = amts[-1]
            else:
                # Then use the product's y-band bounded by neighbouring product rows.
                # This prevents a missing price on one item from stealing the next
                # item's valid right-column price; damaged same-row tokens above
                # already block this fallback.
                selected, amount_source = _amount_for_product_range(
                    r, amount_only, used_amount_line_ids, product_bounds
                )
                if selected is not None and amount_source is not None:
                    used_amount_line_ids.add(str(amount_source))
                    evidence_source = "crop_ocr_right_amount_by_product_band"
        else:
            amount_source = damaged_same_row.get("id")
            evidence_source = "crop_ocr_damaged_amount_blocked_steal"

        if selected is None or selected.get("value") is None:
            # If crop OCR read the product name but damaged/missed the right-column
            # amount, fill from a matching VLM item table row. VLM is not the text
            # authority here; it only supplies an amount for a product row already
            # confirmed by crop OCR.
            vr = _match_vlm_amount(desc, vlm_amount_rows, used_vlm_ids)
            if vr is not None and vr.get("amount") is not None:
                selected = {"value": vr.get("amount"), "raw": vr.get("amount_raw")}
                # Preserve the damaged OCR token as an additional source when it exists.
                source_ids = (
                    [amount_source, vr.get("vlm_row_id")]
                    if amount_source
                    else [vr.get("vlm_row_id")]
                )
                amount_source = vr.get("vlm_row_id")
                used_vlm_ids.add(str(vr.get("vlm_row_id")))
                evidence_source = (
                    "crop_ocr_damaged_amount_vlm_repair"
                    if damaged_same_row is not None
                    else "crop_ocr_product_vlm_amount_fusion"
                )
            else:
                source_ids = [amount_source] if amount_source else []
        else:
            source_ids = [amount_source] if amount_source and amount_source != r.get("id") else []

        if not selected or selected.get("value") is None:
            unmatched_product_rows.append(
                {
                    "row_id": r.get("id"),
                    "text": text,
                    "description_candidate": desc,
                    "reason": "product-like crop OCR row without reliable amount",
                }
            )
            continue

        row = {
            "row_id": r.get("id"),
            "text": text,
            "description_candidate": desc,
            "amount": selected.get("value"),
            "amount_raw": selected.get("raw"),
            "source_line_ids": [r.get("id")]
            + [sid for sid in source_ids if sid and sid != r.get("id")],
            "layout_confidence": round(float(r.get("confidence") or 0.0), 3),
            "evidence_source": evidence_source,
        }
        rows.append(row)

    # Apply conservative final-price grouping after row construction. This turns
    # product original/list price + explicit "Ihr Preis" into one final customer
    # amount before the LLM sees the preferred block.
    rows, final_price_groups = _apply_final_price_overrides(rows, product_lines, final_price_lines)

    # Link quantity/unit-price notes to nearest previous or next item line for LLM support.
    qty_links: list[dict[str, Any]] = []
    for q in quantity_lines:
        qy = float(q.get("y_center") or 0)
        candidates = []
        for row in rows:
            rr = next((x for x in product_lines if x.get("id") == row.get("row_id")), None)
            if rr:
                rr_y = float(rr.get("y_center") or 0)
                ny = next_product_y.get(str(rr.get("id")))
                # Quantity/unit-price notes usually follow the product row and
                # appear before the next product row. Prefer that pattern over
                # pure nearest-y matching at boundaries.
                follows_this_product = qy >= rr_y and (ny is None or qy < ny)
                low, high = product_bounds.get(str(rr.get("id")), (-(10**9), 10**9))
                in_band_bonus = -25 if follows_this_product else (0 if low <= qy < high else 60)
                candidates.append((in_band_bonus + abs(rr_y - qy), row))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            linked = candidates[0][1]
            qty_links.append(
                {
                    "quantity_row_id": q.get("id"),
                    "quantity_text": q.get("text"),
                    "linked_item_row_id": linked.get("row_id"),
                    "linked_item_description": linked.get("description_candidate"),
                    "contributes_hint": "usually_no",
                }
            )

    candidate_sum = round(sum(float(r.get("amount") or 0.0) for r in rows), 2) if rows else None
    printed_total = _find_printed_total(records)
    balanced_to_printed_total = False
    if candidate_sum is not None and printed_total and printed_total.get("amount") is not None:
        balanced_to_printed_total = abs(candidate_sum - float(printed_total["amount"])) <= 0.03
    confidence = "high" if balanced_to_printed_total and rows else ("medium" if rows else "low")
    return {
        "rows": rows[:120],
        "row_count": len(rows),
        "quantity_note_links": qty_links[:80],
        "final_price_adjustment_groups": final_price_groups[:40],
        "unmatched_product_rows": unmatched_product_rows[:50],
        "candidate_sum": candidate_sum,
        "printed_total": printed_total,
        "balanced_to_printed_total": balanced_to_printed_total,
        "confidence": confidence,
        "method": "vlm_region_bbox_crop_reocr_conservative_amounts_v14_13_3",
        "item_zone": {"start_y": start_y, "stop_y": boundary_y, "record_count": len(item_zone)},
        "vlm_amount_fusion_count": sum(
            1 for r in rows if r.get("evidence_source") == "crop_ocr_product_vlm_amount_fusion"
        ),
    }


def run_vlm_region_reocr(
    *,
    source_image_path: Path,
    vlm_result: dict[str, Any],
    visual_evidence: dict[str, Any] | None = None,
    result_dir: Path,
    run_id: str,
    lang: str = "german",
    device: str = "cpu",
    max_regions: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_image_path = Path(source_image_path)
    out_dir = result_dir / "region_reocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not source_image_path.exists():
        return {
            "status": "error",
            "error": f"source image missing: {source_image_path}",
            "duration_seconds": 0.0,
        }
    if source_image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return {
            "status": "skipped",
            "error": f"unsupported image extension: {source_image_path.suffix}",
            "duration_seconds": 0.0,
        }

    prep = _image_prepare(vlm_result, source_image_path)
    layout_boxes = _extract_layout_boxes(vlm_result)
    selected = _select_region_boxes(layout_boxes, prep)[:max_regions]
    regions: list[dict[str, Any]] = []
    preferred_blocks: list[dict[str, Any]] = []
    for idx, box in enumerate(selected):
        bbox_original = _map_bbox_to_original(box.get("coordinate") or [], prep, pad=24)
        crop_path = out_dir / f"{run_id}_region_{idx:02d}_{box.get('kind') or 'region'}.jpg"
        crop_info = _crop_image(source_image_path, bbox_original, crop_path)
        try:
            ocr = _run_paddle_ocr_v13_profile(
                crop_path,
                lang=lang,
                device=device,
                min_score=0.20,
                text_detection_model_name=None,
                text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
            )
            lines = _line_records_from_ocr(
                ocr, crop_bbox=bbox_original, crop_scale=float(crop_info.get("crop_scale") or 1.0)
            )
            block = _build_preferred_item_block(lines, visual_evidence=visual_evidence)
            block["region_id"] = f"region_{idx:02d}"
            block["crop_path"] = str(crop_path)
            block["bbox_original"] = bbox_original
            preferred_blocks.append(block)
            region = {
                **box,
                "region_id": f"region_{idx:02d}",
                "bbox_original": bbox_original,
                "crop": crop_info,
                "ocr_status": "ok",
                "ocr_word_count": len(ocr.get("words") or []),
                "lines": lines[:200],
                "preferred_item_block": block,
            }
        except Exception as exc:
            region = {
                **box,
                "region_id": f"region_{idx:02d}",
                "bbox_original": bbox_original,
                "crop": crop_info,
                "ocr_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        regions.append(region)

    best_block = None
    if preferred_blocks:
        preferred_blocks.sort(
            key=lambda b: (
                1 if b.get("balanced_to_printed_total") else 0,
                b.get("row_count") or 0,
                b.get("candidate_sum") or 0,
            ),
            reverse=True,
        )
        best_block = preferred_blocks[0]

    return {
        "schema_version": "v14_13_region_reocr_3",
        "status": "ok" if regions else "no_regions",
        "source_image_path": str(source_image_path),
        "image_prepare": prep,
        "layout_box_count": len(layout_boxes),
        "selected_region_count": len(selected),
        "selected_regions": [
            {k: v for k, v in r.items() if k not in {"lines", "preferred_item_block"}}
            for r in regions
        ],
        "regions": regions,
        "preferred_item_blocks": preferred_blocks[:3],
        "best_preferred_item_block": best_block,
        "duration_seconds": round(time.perf_counter() - started, 2),
    }


def merge_region_reocr_into_visual_evidence(
    visual_evidence: dict[str, Any] | None, region_reocr: dict[str, Any]
) -> dict[str, Any]:
    evidence = dict(visual_evidence or {"status": "ok", "backend": "region_reocr_only"})
    evidence["status"] = evidence.get("status") or "ok"
    evidence["region_reocr"] = region_reocr
    blocks = region_reocr.get("preferred_item_blocks") or []
    if blocks:
        evidence["preferred_item_blocks"] = (evidence.get("preferred_item_blocks") or []) + blocks
        region_final_groups = []
        for b in blocks:
            if isinstance(b, dict):
                region_final_groups.extend(b.get("final_price_adjustment_groups") or [])
        if region_final_groups:
            evidence["final_price_adjustment_groups"] = region_final_groups + (
                evidence.get("final_price_adjustment_groups") or []
            )
    best = region_reocr.get("best_preferred_item_block")
    if best:
        evidence["best_preferred_item_block"] = best
    summary = dict(evidence.get("summary") or {})
    summary.update(
        {
            "region_reocr_status": region_reocr.get("status"),
            "region_reocr_selected_region_count": region_reocr.get("selected_region_count"),
            "preferred_item_block_count": len(blocks),
            "best_preferred_item_block_balanced": bool(
                best and best.get("balanced_to_printed_total")
            ),
            "best_preferred_item_block_sum": best.get("candidate_sum")
            if isinstance(best, dict)
            else None,
            "best_preferred_item_block_total": (best.get("printed_total") or {}).get("amount")
            if isinstance(best, dict) and isinstance(best.get("printed_total"), dict)
            else None,
        }
    )
    evidence["summary"] = summary
    guidance = list(evidence.get("semantic_guidance") or [])
    guidance.insert(
        0,
        "V14.13.3: Preferred item blocks from VLM-region crop re-OCR are primary item evidence when their candidate sum reconciles to the printed total.",
    )
    guidance.insert(
        1,
        "VLM text is not final truth; VLM bboxes locate regions, crop OCR reads exact rows, and geometry aligns item names with right-column prices.",
    )
    evidence["semantic_guidance"] = guidance[:12]
    return evidence
