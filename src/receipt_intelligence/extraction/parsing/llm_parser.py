#!/usr/bin/env python3
"""
LLM receipt parser with VLM-region-first crop re-OCR and consistency evidence.

Architecture:
    OCR JSON -> compact coordinate/layout context -> local LLM extracts full receipt JSON

This module deliberately does NOT create deterministic item rows as a fallback.
Deterministic code is allowed only to:
    - prepare OCR/layout evidence for the prompt
    - coerce/validate JSON shape
    - record why the LLM call failed

Expected OCR input:
    A JSON object with image_width/image_height and either:
      - words: [{text, confidence, xmin, ymin, xmax, ymax, ...}, ...]
      - lines: [{text, confidence, bbox/xmin/ymin/xmax/ymax, ...}, ...]
    The PaddleOCR app format used in earlier versions is supported.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    GenerationResult,
    LlmGateway,
    coerce_generation_result,
)
from receipt_intelligence.extraction.evidence.compact import (
    build_compact_evidence,
    compact_evidence_to_prompt_text,
)
from receipt_intelligence.extraction.evidence.layout import (
    build_layout_context,
    extract_ocr_amounts,
)
from receipt_intelligence.extraction.evidence.visual import visual_evidence_to_prompt_text
from receipt_intelligence.prompts import render_prompt_template


# Accept JSON/model amount strings with decimal comma or decimal dot, but OCR
# amount extraction below uses extract_ocr_amounts() to avoid parsing dates like
# 09.12 as money.
AMOUNT_RE = re.compile(
    r"(?<!\d)([-+−]?\s*\d{1,5}(?:[.\s]\d{3})*(?:[,.]\s*\d{1,2})|[-+−]?\s*\d{1,5}\s+\d{2})(?:\s*[-−])?(?!\d)"
)
DATE_RE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
FOOTER_HINT_RE = re.compile(
    r"\b(SUMME|TOTAL|GESAMT|BETRAG|MWST|UST|NETTO|BRUTTO|STEUER|EC|GIROCARD|KARTE|VISA|MASTERCARD|BAR|CASH|GEGEBEN|RÜCKGELD|RUECKGELD|DATUM|UHRZEIT|BELEG|BON|TRACE|TERMINAL|KUNDENBELEG)\b",
    re.IGNORECASE,
)


@dataclass
class OcrWord:
    id: str
    text: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def x_center(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def y_center(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)


@dataclass
class OcrLine:
    line_id: str
    text: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    source_word_ids: list[str]
    left_text: str = ""
    right_text: str = ""
    amount_candidates: list[dict[str, Any]] | None = None
    flags: list[str] | None = None

    def compact(self, image_width: int, image_height: int) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x": round(self.xmin / max(image_width, 1), 4),
                "y": round(self.ymin / max(image_height, 1), 4),
                "w": round((self.xmax - self.xmin) / max(image_width, 1), 4),
                "h": round((self.ymax - self.ymin) / max(image_height, 1), 4),
            },
            "left_text": self.left_text,
            "right_text": self.right_text,
            "amount_candidates": self.amount_candidates or [],
            "flags": self.flags or [],
            "source_word_ids": self.source_word_ids,
        }


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def parse_amount(text: Any) -> float | None:
    """Parse model/JSON numeric values into money-like floats.

    OCR evidence uses extract_amounts() below, which intentionally requires
    decimal comma to avoid reading dates such as 09.12 as money. This function
    is more permissive because the LLM may return JSON strings like "47.45".
    """
    if text is None or isinstance(text, bool):
        return None
    if isinstance(text, (int, float)) and math.isfinite(float(text)):
        return round(float(text), 2)
    raw = str(text).strip().replace("−", "-")
    if not raw:
        return None
    # Date/time-only strings are not amounts.
    if (DATE_RE.search(raw) or TIME_RE.search(raw)) and "," not in raw:
        return None
    m = AMOUNT_RE.search(raw)
    if not m:
        return None
    token = m.group(0).strip()
    negative = (
        token.startswith("-")
        or token.endswith("-")
        or re.search(r"[,\.]\s*\d{1,2}\s*[-−]", token) is not None
    )
    s = re.sub(r"[^0-9,\.\s]", "", token).strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        # Last separator is decimal separator.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "." in s:
        # Decimal dot is allowed only for normalized LLM strings, not date-like
        # OCR lines. Already excluded date/time above.
        pass
    elif re.fullmatch(r"\d{3,7}", s):
        return None
    try:
        value = round(float(s), 2)
    except Exception:
        return None
    return -abs(value) if negative else value


def extract_amounts(text: str) -> list[dict[str, Any]]:
    """Extract OCR money candidates safely.

    This uses the V14.1 layout helper and intentionally avoids dot-decimal OCR
    candidates, preventing dates such as 09.12.17 from becoming 9.12 EUR.
    """
    return extract_ocr_amounts(text or "")


def _candidate_words_from_raw(
    raw_words: list[Any], image_width: int, image_height: int
) -> list[OcrWord]:
    words: list[OcrWord] = []
    for idx, raw in enumerate(raw_words):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        # Paddle app variants may use either confidence or score.
        conf = _to_float(raw.get("confidence", raw.get("score", 0.0)), 0.0)
        bbox = raw.get("bbox") if isinstance(raw.get("bbox"), dict) else None
        if all(k in raw for k in ("xmin", "ymin", "xmax", "ymax")):
            xmin = _to_float(raw.get("xmin"))
            ymin = _to_float(raw.get("ymin"))
            xmax = _to_float(raw.get("xmax"))
            ymax = _to_float(raw.get("ymax"))
        elif bbox and all(k in bbox for k in ("x", "y", "w", "h")):
            x = _to_float(bbox.get("x"))
            y = _to_float(bbox.get("y"))
            w = _to_float(bbox.get("w"))
            h = _to_float(bbox.get("h"))
            # bbox may already be normalized.
            if x <= 1.5 and y <= 1.5 and w <= 1.5 and h <= 1.5:
                xmin, ymin = x * image_width, y * image_height
                xmax, ymax = (x + w) * image_width, (y + h) * image_height
            else:
                xmin, ymin, xmax, ymax = x, y, x + w, y + h
        elif isinstance(raw.get("polygon"), list) and raw.get("polygon"):
            pts = raw["polygon"]
            xs = [_to_float(p[0]) for p in pts if isinstance(p, list) and len(p) >= 2]
            ys = [_to_float(p[1]) for p in pts if isinstance(p, list) and len(p) >= 2]
            if not xs or not ys:
                continue
            xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        words.append(
            OcrWord(
                id=str(raw.get("id") or raw.get("word_id") or f"word_{idx:04d}"),
                text=text,
                confidence=conf,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )
    return words


def _extract_words(data: dict[str, Any]) -> tuple[list[OcrWord], int, int]:
    image_width = int(data.get("image_width") or data.get("width") or 1)
    image_height = int(data.get("image_height") or data.get("height") or 1)
    raw_words = data.get("words")
    if raw_words is None and isinstance(data.get("ocr"), dict):
        raw_words = data["ocr"].get("words")
    if not isinstance(raw_words, list):
        raw_words = []
    words = _candidate_words_from_raw(raw_words, image_width, image_height)
    if words and (image_width <= 1 or image_height <= 1):
        image_width = int(max(w.xmax for w in words) + 1)
        image_height = int(max(w.ymax for w in words) + 1)
    return words, image_width, image_height


def _line_from_words(line_id: str, line_words: list[OcrWord], image_width: int) -> OcrLine:
    ordered = sorted(line_words, key=lambda w: (w.xmin, w.ymin))
    text = " ".join(w.text for w in ordered).strip()
    xmin = min(w.xmin for w in ordered)
    ymin = min(w.ymin for w in ordered)
    xmax = max(w.xmax for w in ordered)
    ymax = max(w.ymax for w in ordered)
    conf = sum(w.confidence for w in ordered) / max(len(ordered), 1)
    left = " ".join(w.text for w in ordered if w.x_center < image_width * 0.58).strip()
    right = " ".join(w.text for w in ordered if w.x_center >= image_width * 0.58).strip()
    amounts = extract_amounts(text)
    flags: list[str] = []
    if amounts:
        flags.append("has_amount")
    if DATE_RE.search(text):
        flags.append("date_candidate")
    if TIME_RE.search(text):
        flags.append("time_candidate")
    if FOOTER_HINT_RE.search(text):
        flags.append("footer_or_total_payment_tax_hint")
    if conf < 0.75:
        flags.append("low_confidence")
    return OcrLine(
        line_id=line_id,
        text=text,
        confidence=conf,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        source_word_ids=[w.id for w in ordered],
        left_text=left,
        right_text=right,
        amount_candidates=amounts,
        flags=flags,
    )


def _group_words_into_lines(
    words: list[OcrWord], image_width: int, image_height: int
) -> list[OcrLine]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.y_center, w.xmin))
    heights = sorted(w.height for w in ordered if w.height > 0)
    median_height = heights[len(heights) // 2] if heights else max(image_height * 0.01, 10)
    threshold = max(8.0, median_height * 0.65)
    clusters: list[list[OcrWord]] = []
    for w in ordered:
        placed = False
        for cluster in clusters[-3:]:
            cy = sum(x.y_center for x in cluster) / len(cluster)
            if abs(w.y_center - cy) <= threshold:
                cluster.append(w)
                placed = True
                break
        if not placed:
            clusters.append([w])
    lines = [
        _line_from_words(f"line_{i:03d}", cluster, image_width)
        for i, cluster in enumerate(clusters)
    ]
    return sorted(lines, key=lambda line: (line.ymin, line.xmin))


def _lines_from_existing(
    data: dict[str, Any], image_width: int, image_height: int
) -> list[OcrLine]:
    raw_lines = data.get("lines")
    if raw_lines is None and isinstance(data.get("ocr"), dict):
        raw_lines = data["ocr"].get("lines")
    if not isinstance(raw_lines, list):
        return []
    out: list[OcrLine] = []
    for idx, raw in enumerate(raw_lines):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        bbox = raw.get("bbox") if isinstance(raw.get("bbox"), dict) else None
        if all(k in raw for k in ("xmin", "ymin", "xmax", "ymax")):
            xmin = _to_float(raw.get("xmin"))
            ymin = _to_float(raw.get("ymin"))
            xmax = _to_float(raw.get("xmax"))
            ymax = _to_float(raw.get("ymax"))
        elif bbox and all(k in bbox for k in ("x", "y", "w", "h")):
            x = _to_float(bbox.get("x"))
            y = _to_float(bbox.get("y"))
            w = _to_float(bbox.get("w"))
            h = _to_float(bbox.get("h"))
            if x <= 1.5 and y <= 1.5 and w <= 1.5 and h <= 1.5:
                xmin, ymin = x * image_width, y * image_height
                xmax, ymax = (x + w) * image_width, (y + h) * image_height
            else:
                xmin, ymin, xmax, ymax = x, y, x + w, y + h
        else:
            # Preserve existing line even if no bbox exists.
            xmin, ymin, xmax, ymax = 0.0, idx * 10.0, float(image_width), idx * 10.0 + 10.0
        amounts = extract_amounts(text)
        flags: list[str] = []
        if amounts:
            flags.append("has_amount")
        if DATE_RE.search(text):
            flags.append("date_candidate")
        if TIME_RE.search(text):
            flags.append("time_candidate")
        if FOOTER_HINT_RE.search(text):
            flags.append("footer_or_total_payment_tax_hint")
        conf = _to_float(raw.get("confidence", 0.0), 0.0)
        if conf < 0.75:
            flags.append("low_confidence")
        out.append(
            OcrLine(
                line_id=str(raw.get("line_id") or raw.get("id") or f"line_{idx:03d}"),
                text=text,
                confidence=conf,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                source_word_ids=[str(x) for x in raw.get("source_word_ids") or []],
                left_text=str(raw.get("left_text") or ""),
                right_text=str(raw.get("right_text") or ""),
                amount_candidates=amounts,
                flags=flags,
            )
        )
    return sorted(out, key=lambda line: (line.ymin, line.xmin))


def build_ocr_context(data: dict[str, Any], max_lines: int = 260) -> dict[str, Any]:
    words, image_width, image_height = _extract_words(data)
    lines = _lines_from_existing(data, image_width, image_height)
    if not lines:
        lines = _group_words_into_lines(words, image_width, image_height)

    # Keep line ids stable after sorting.
    for idx, line in enumerate(lines):
        if not line.line_id:
            line.line_id = f"line_{idx:03d}"

    if max_lines and len(lines) > max_lines:
        # Keep top and bottom; totals/payment/date are often near footer.
        head_count = max_lines // 2
        tail_count = max_lines - head_count
        kept = lines[:head_count] + lines[-tail_count:]
        omitted = len(lines) - len(kept)
    else:
        kept = lines
        omitted = 0

    compact_lines = [line.compact(image_width, image_height) for line in kept]
    plain_text = "\n".join(f"[{line.line_id}] {line.text}" for line in kept)
    table_view = "\n".join(
        f"[{line.line_id}] {line.left_text[:54]:54s} | {line.right_text[:34]}"
        if (line.left_text or line.right_text)
        else f"[{line.line_id}] {line.text}"
        for line in kept
    )
    amount_lines = [
        {
            "line_id": line.line_id,
            "text": line.text,
            "amount_candidates": line.amount_candidates or [],
            "flags": line.flags or [],
        }
        for line in kept
        if line.amount_candidates
    ]
    low_confidence_lines = [
        {"line_id": line.line_id, "text": line.text, "confidence": round(line.confidence, 4)}
        for line in kept
        if line.confidence and line.confidence < 0.75
    ]
    date_time_candidates = [
        {"line_id": line.line_id, "text": line.text}
        for line in kept
        if DATE_RE.search(line.text) or TIME_RE.search(line.text)
    ]
    layout_context = build_layout_context(
        compact_lines, image_width=image_width, image_height=image_height
    )

    return {
        "schema_version": "v14_6_ocr_context_1",
        "image_width": image_width,
        "image_height": image_height,
        "word_count": len(words),
        "line_count": len(lines),
        "kept_line_count": len(kept),
        "omitted_middle_line_count": omitted,
        "plain_text": plain_text,
        "table_view": table_view,
        "layout_view": layout_context.get("layout_view", ""),
        "lines": compact_lines,
        "layout_rows": layout_context.get("aligned_rows", []),
        "amount_neighbors": layout_context.get("amount_neighbors", []),
        "semantic_candidate_hints": layout_context.get("semantic_candidate_hints", []),
        "layout_context": layout_context,
        "amount_lines": amount_lines,
        "low_confidence_lines": low_confidence_lines,
        "date_time_candidates": date_time_candidates,
    }


def _visual_evidence_line_registry(
    visual_evidence: dict[str, Any] | None, limit: int = 800
) -> list[dict[str, Any]]:
    """Flatten VLM/region evidence rows into an ID registry for validation.

    The LLM may cite region_line_*, vlm_table_*_row_*, or preferred item
    source ids. These IDs are valid evidence even though they are not full-image
    OCR line ids.
    """
    if not isinstance(visual_evidence, dict):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        lid: Any, text: Any = None, confidence: Any = 1.0, amounts: Any = None, **extra: Any
    ) -> None:
        if len(out) >= limit:
            return
        if lid is None or not str(lid).strip():
            return
        sid = str(lid)
        if sid in seen:
            return
        seen.add(sid)
        out.append(
            {
                "line_id": sid,
                "text": str(text or "visual evidence row"),
                "confidence": confidence,
                "amount_candidates": amounts or [],
                **extra,
            }
        )

    for table in visual_evidence.get("structured_tables") or []:
        for row in table.get("rows") or []:
            if isinstance(row, dict):
                cells = row.get("cells") or []
                add(
                    row.get("id"),
                    " | ".join(str(c) for c in cells) or row.get("text"),
                    1.0,
                    row.get("amounts"),
                    evidence_type="vlm_table_row",
                )

    region_reocr = (
        visual_evidence.get("region_reocr")
        if isinstance(visual_evidence.get("region_reocr"), dict)
        else {}
    )
    for region in region_reocr.get("regions") or []:
        if not isinstance(region, dict):
            continue
        rid = region.get("region_id")
        for line in region.get("lines") or []:
            if isinstance(line, dict):
                add(
                    line.get("id"),
                    line.get("text"),
                    line.get("confidence", line.get("layout_confidence", 1.0)),
                    line.get("amounts"),
                    evidence_type="region_crop_ocr_line",
                    region_id=rid,
                )

    for block in visual_evidence.get("preferred_item_blocks") or []:
        if not isinstance(block, dict):
            continue
        for row in block.get("rows") or []:
            if not isinstance(row, dict):
                continue
            add(
                row.get("row_id"),
                row.get("text") or row.get("description_candidate"),
                row.get("layout_confidence", 1.0),
                (
                    [{"raw": row.get("amount_raw"), "value": row.get("amount")}]
                    if row.get("amount") is not None
                    else []
                ),
                evidence_type="preferred_item_block_row",
                source_line_ids=row.get("source_line_ids") or [],
            )
            for sid in row.get("source_line_ids") or []:
                add(
                    sid,
                    row.get("text") or row.get("description_candidate"),
                    row.get("layout_confidence", 1.0),
                    (
                        [{"raw": row.get("amount_raw"), "value": row.get("amount")}]
                        if row.get("amount") is not None
                        else []
                    ),
                    evidence_type="preferred_item_source",
                )
        pt = block.get("printed_total") if isinstance(block.get("printed_total"), dict) else None
        if pt:
            add(
                pt.get("row_id"),
                pt.get("text") or "printed total",
                1.0,
                (
                    [{"raw": pt.get("raw"), "value": pt.get("amount")}]
                    if pt.get("amount") is not None
                    else []
                ),
                evidence_type="preferred_item_printed_total",
            )
            add(
                pt.get("amount_row_id"),
                pt.get("text") or "printed total amount",
                1.0,
                (
                    [{"raw": pt.get("raw"), "value": pt.get("amount")}]
                    if pt.get("amount") is not None
                    else []
                ),
                evidence_type="preferred_item_printed_total_amount",
            )

    return out


def receipt_schema_for_prompt() -> dict[str, Any]:
    return {
        "schema_version": "v14_6_llm_receipt_1",
        "parse_status": "ok | partial | failed",
        "currency": "EUR or null",
        "merchant": {
            "name": "string or null",
            "address": "string or null",
            "tax_id": "string or null",
            "source_line_ids": ["line_000"],
        },
        "date": "YYYY-MM-DD or null",
        "time": "HH:MM:SS or HH:MM or null",
        "items": [
            {
                "raw_description": "string or null; full row text as printed when available",
                "description": "string; clean product/row description used by legacy code",
                "product_description": "string or null; product-identifying text only, without coupon/promotion/context note when separable",
                "line_note": "string or null; non-product context printed with row, e.g. coupon/promotion note",
                "promotion_note": "string or null; coupon/discount/promotion note if separable",
                "quantity": "number or null",
                "unit": "string or null",
                "unit_price": "number or null",
                "original_price": "number or null, optional when receipt shows original/list price before discount",
                "discount_amount": "number or null, optional when receipt has an item-level discount column",
                "line_total": "number or null",
                "tax_rate": "number or null",
                "tax_code": "string or null, optional suffix/marker such as a/b/A/B",
                "category": "item | discount | deposit | refund | unknown",
                "source_line_ids": ["line_000"],
                "table_interpretation_source_row_id": "string or null, optional",
                "confidence": "0..1",
                "notes": "string or null",
            }
        ],
        "taxes": [
            {
                "rate": "number or null",
                "net": "number or null",
                "tax": "number or null",
                "gross": "number or null",
                "source_line_ids": ["line_000"],
            }
        ],
        "totals": {
            "subtotal": "number or null",
            "tax_total": "number or null",
            "grand_total": "number or null",
            "paid_total": "number or null",
            "change": "number or null",
            "source_line_ids": ["line_000"],
        },
        "payments": [
            {
                "method": "cash | girocard | credit_card | debit_card | voucher | coupon | gift_card | unknown",
                "amount": "number or null",
                "source_line_ids": ["line_000"],
            }
        ],
        "unresolved_rows": [
            {
                "line_id": "line_000",
                "text": "OCR row that may contain useful information but could not be classified safely",
                "reason": "string",
            }
        ],
        "warnings": ["string"],
        "overall_confidence": "0..1",
    }


def _limit_list(values: Any, limit: int) -> list[Any]:
    if not isinstance(values, list):
        return []
    return values[: max(0, limit)]


def _compact_schema_for_prompt() -> dict[str, Any]:
    """Smaller schema for retry prompts; keeps the exact same root shape."""
    return {
        "schema_version": "v14_6_llm_receipt_1",
        "parse_status": "ok|partial|failed",
        "currency": "EUR|null",
        "merchant": {"name": None, "address": None, "tax_id": None, "source_line_ids": []},
        "date": "YYYY-MM-DD|null",
        "time": "HH:MM:SS|null",
        "items": [
            {
                "raw_description": None,
                "description": "string",
                "product_description": None,
                "line_note": None,
                "promotion_note": None,
                "quantity": None,
                "unit": None,
                "unit_price": None,
                "original_price": None,
                "discount_amount": None,
                "line_total": None,
                "tax_rate": None,
                "tax_code": None,
                "category": "item|discount|deposit|refund|unknown",
                "source_line_ids": [],
                "table_interpretation_source_row_id": None,
                "confidence": 0.0,
                "notes": None,
            }
        ],
        "taxes": [{"rate": None, "net": None, "tax": None, "gross": None, "source_line_ids": []}],
        "totals": {
            "subtotal": None,
            "tax_total": None,
            "grand_total": None,
            "paid_total": None,
            "change": None,
            "source_line_ids": [],
        },
        "payments": [
            {
                "method": "cash|girocard|credit_card|debit_card|voucher|coupon|gift_card|unknown",
                "amount": None,
                "source_line_ids": [],
            }
        ],
        "unresolved_rows": [],
        "warnings": [],
        "overall_confidence": 0.0,
    }


def _compact_layout_rows(rows: list[dict[str, Any]], limit: int = 120) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "row_id": row.get("row_id"),
                "left_text": row.get("left_text"),
                "right_text": row.get("right_text"),
                "right_amounts": row.get("right_amounts") or row.get("amounts") or [],
                "source_line_ids": row.get("source_line_ids") or [],
            }
        )
    return out


def _compact_lines(lines: list[dict[str, Any]], limit: int = 120) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines[:limit]:
        if not isinstance(line, dict):
            continue
        out.append(
            {
                "line_id": line.get("line_id"),
                "text": line.get("text"),
                "amounts": line.get("amount_candidates") or [],
                "flags": line.get("flags") or [],
            }
        )
    return out


def _strict_output_schema_for_prompt() -> dict[str, Any]:
    """Exact V14.6 schema shown to the model. Keep this compact."""
    return {
        "schema_version": "v14_6_llm_receipt_1",
        "parse_status": "ok|partial|failed",
        "currency": "EUR|null",
        "merchant": {"name": None, "address": None, "tax_id": None, "source_line_ids": []},
        "date": "YYYY-MM-DD|null",
        "time": "HH:MM:SS|null",
        "items": [
            {
                "raw_description": None,
                "description": "string",
                "product_description": None,
                "line_note": None,
                "promotion_note": None,
                "quantity": None,
                "unit": None,
                "unit_price": None,
                "original_price": None,
                "discount_amount": None,
                "line_total": None,
                "tax_rate": None,
                "tax_code": None,
                "category": "item|discount|deposit|refund|unknown",
                "source_line_ids": [],
                "table_interpretation_source_row_id": None,
                "confidence": 0.0,
                "notes": None,
            }
        ],
        "taxes": [{"rate": None, "net": None, "tax": None, "gross": None, "source_line_ids": []}],
        "totals": {
            "subtotal": None,
            "tax_total": None,
            "grand_total": None,
            "paid_total": None,
            "change": None,
            "source_line_ids": [],
        },
        "payments": [
            {
                "method": "cash|girocard|credit_card|debit_card|voucher|coupon|gift_card|unknown",
                "amount": None,
                "source_line_ids": [],
            }
        ],
        "unresolved_rows": [{"line_id": "line_000", "text": "string", "reason": "string"}],
        "warnings": [],
        "overall_confidence": 0.0,
    }


def build_prompt(
    ocr_context: dict[str, Any],
    prompt_profile: str = "compact_evidence",
    previous_error: str | None = None,
    visual_evidence: dict[str, Any] | None = None,
) -> str:
    """Build the V14.6 compact-evidence LLM prompt.

    V14.6 intentionally does not send bbox/coordinate JSON or the full OCR context
    to the model. The model gets receipt-shaped evidence only: header candidates,
    reconstructed rows, candidate groups, neighbor context, and minimal raw text.
    """
    retry_note = ""
    has_visual = bool(
        visual_evidence and visual_evidence.get("status") in {"ok", "no_amounts_found"}
    )
    # When VLM/table evidence exists, the OCR evidence becomes secondary and can be shorter.
    # This avoids the previous long/noisy OCR-only prompt that caused slow or truncated JSON.
    max_rows = 90 if has_visual else 140
    if prompt_profile in {"schema_retry", "wrong_schema_retry"}:
        retry_note = (
            "\nIMPORTANT RETRY: your previous answer was not accepted. "
            "Return the exact schema below. Do not summarize, do not echo OCR input, "
            "and do not output box_2d/bbox/object-detection JSON.\n"
        )
        max_rows = 70 if has_visual else 120
    elif prompt_profile == "ultra_compact":
        retry_note = "\nULTRA-COMPACT RETRY: output only the required receipt JSON schema.\n"
        max_rows = 45 if has_visual else 90

    evidence = build_compact_evidence(ocr_context, max_rows=max_rows)
    evidence_text = compact_evidence_to_prompt_text(evidence)
    visual_text = visual_evidence_to_prompt_text(visual_evidence) if has_visual else ""
    schema = _strict_output_schema_for_prompt()
    error_block = f"\nPrevious rejection reason: {previous_error}\n" if previous_error else ""

    return render_prompt_template(
        "main_receipt_parser.txt",
        RETRY_NOTE=retry_note,
        ERROR_BLOCK=error_block,
        SCHEMA_JSON=json.dumps(schema, ensure_ascii=False, indent=2),
        VISUAL_TEXT=visual_text,
        EVIDENCE_TEXT=evidence_text,
    )


def ollama_generate(
    *,
    ollama_url: str,
    model: str,
    prompt: str,
    num_ctx: int = 24384,
    num_predict: int = 8192,
    temperature: float = 0.0,
    keep_alive: str | None = None,
    timeout: float = 240.0,
    format_json: bool = True,
) -> GenerationResult:
    """Compatibility entry point; new code injects ``LlmGateway`` directly."""

    from receipt_intelligence.adapters.llm import OllamaGateway

    return OllamaGateway(ollama_url).generate(
        GenerationRequest(
            model=model,
            prompt=prompt,
            operation="legacy_receipt_generation",
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=temperature,
            keep_alive=keep_alive,
            timeout_seconds=timeout,
            format_json=format_json,
        )
    )


class LLMWrongSchemaError(ValueError):
    """Raised when the model returns JSON that is not the receipt schema."""


EXPECTED_SCHEMA_VERSION = "v14_6_llm_receipt_1"
SCHEMA_VERSION_ALIASES = {
    "v14_6_llm_llm_receipt_1",
    "v14_4_llm_receipt_1",
    "v14_4_llm_llm_receipt_1",
    "v14_3_llm_receipt_1",
    "v14_3_llm_llm_receipt_1",
}


def _required_schema_keys() -> set[str]:
    """Top-level fields whose absence makes a receipt structurally unusable.

    ``overall_confidence`` is intentionally not included.  It is advisory
    metadata and can be normalized locally without spending another full LLM
    generation on an otherwise valid receipt.
    """
    return {
        "schema_version",
        "parse_status",
        "currency",
        "merchant",
        "date",
        "time",
        "items",
        "taxes",
        "totals",
        "payments",
        "unresolved_rows",
        "warnings",
    }


def normalize_noncritical_schema_defaults(obj: dict[str, Any]) -> dict[str, Any]:
    """Fill advisory schema fields that do not justify a complete LLM retry."""
    if not isinstance(obj, dict):
        return obj
    fixed = dict(obj)
    if "overall_confidence" not in fixed or fixed.get("overall_confidence") is None:
        fixed["overall_confidence"] = 0.6
        warnings = fixed.get("warnings")
        if isinstance(warnings, list):
            fixed["warnings"] = list(warnings) + [
                "Missing overall_confidence normalized to 0.6 without an LLM retry."
            ]
    return fixed


def normalize_schema_version_alias(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize harmless schema-version typos only.

    This is not semantic fallback: it changes only the version string when the
    model returned the full required receipt object but made a known typo such
    as v14_6_llm_llm_receipt_1.
    """
    if not isinstance(obj, dict):
        return obj
    version = str(obj.get("schema_version") or "")
    if version == EXPECTED_SCHEMA_VERSION:
        return obj
    if version in SCHEMA_VERSION_ALIASES and _required_schema_keys().issubset(set(obj.keys())):
        fixed = dict(obj)
        fixed["schema_version"] = EXPECTED_SCHEMA_VERSION
        fixed.setdefault("warnings", [])
        if isinstance(fixed["warnings"], list):
            fixed["warnings"] = list(fixed["warnings"]) + [
                f"Schema version normalized from {version} to {EXPECTED_SCHEMA_VERSION}."
            ]
        return fixed
    return obj


def validate_llm_receipt_schema_object(obj: dict[str, Any]) -> None:
    """Reject valid JSON that is not the V14.6 receipt schema.

    This prevents accepting summaries, OCR-context echoes, and object-detection
    outputs such as {"box_2d": ..., "label": ...}. Harmless schema-version
    typos are normalized before this validation in run_llm_main_parser().
    """
    if not isinstance(obj, dict):
        raise LLMWrongSchemaError("JSON root is not an object")
    forbidden_keys = {
        "box_2d",
        "bbox",
        "label",
        "text_content",
        "layout_rows",
        "plain_text",
        "lines",
        "image_width",
        "image_height",
    }
    found_forbidden = sorted(k for k in forbidden_keys if k in obj)
    if found_forbidden:
        raise LLMWrongSchemaError(
            f"wrong schema: forbidden OCR/object-detection keys present: {found_forbidden}"
        )
    if obj.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise LLMWrongSchemaError(
            f"wrong schema_version: expected {EXPECTED_SCHEMA_VERSION!r}, got {obj.get('schema_version')!r}"
        )
    missing = sorted(_required_schema_keys() - set(obj.keys()))
    if missing:
        raise LLMWrongSchemaError(f"wrong schema: missing required top-level keys: {missing}")
    if not isinstance(obj.get("merchant"), dict):
        raise LLMWrongSchemaError("wrong schema: merchant must be an object")
    if not isinstance(obj.get("totals"), dict):
        raise LLMWrongSchemaError("wrong schema: totals must be an object")
    for key in ("items", "taxes", "payments", "unresolved_rows", "warnings"):
        if not isinstance(obj.get(key), list):
            raise LLMWrongSchemaError(f"wrong schema: {key} must be an array")
    required_totals = {
        "subtotal",
        "tax_total",
        "grand_total",
        "paid_total",
        "change",
        "source_line_ids",
    }
    missing_totals = sorted(required_totals - set(obj.get("totals", {}).keys()))
    if missing_totals:
        raise LLMWrongSchemaError(f"wrong schema: totals missing keys: {missing_totals}")


def _num_or_none(value: Any) -> float | None:
    return parse_amount(value)


def _list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None and str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_llm_receipt(obj: dict[str, Any]) -> dict[str, Any]:
    """Light schema coercion only. This does not create deterministic content."""
    receipt: dict[str, Any] = dict(obj)
    receipt["schema_version"] = str(receipt.get("schema_version") or "v14_6_llm_receipt_1")
    if receipt.get("parse_status") not in {"ok", "partial", "failed"}:
        receipt["parse_status"] = "partial"
    receipt["currency"] = receipt.get("currency") or "EUR"

    merchant = receipt.get("merchant") if isinstance(receipt.get("merchant"), dict) else {}
    receipt["merchant"] = {
        "name": merchant.get("name"),
        "address": merchant.get("address"),
        "tax_id": merchant.get("tax_id"),
        "source_line_ids": _list_of_str(merchant.get("source_line_ids")),
    }

    items: list[dict[str, Any]] = []
    for raw in receipt.get("items") or []:
        if not isinstance(raw, dict):
            continue
        desc = raw.get("description") or raw.get("name")
        if desc is None and raw.get("line_total") is None:
            continue
        item = dict(raw)
        item["raw_description"] = (
            str(item.get("raw_description") or item.get("raw_text") or desc or "").strip() or None
        )
        product_desc = item.get("product_description") or item.get("clean_description") or desc
        item["product_description"] = (
            str(product_desc).strip() if product_desc is not None else None
        )
        item["description"] = (
            str(item.get("description") or product_desc or desc).strip()
            if (item.get("description") is not None or product_desc is not None or desc is not None)
            else None
        )
        item["line_note"] = (
            str(item.get("line_note") or item.get("context_note") or "").strip() or None
        )
        item["promotion_note"] = (
            str(item.get("promotion_note") or item.get("discount_note") or "").strip() or None
        )
        item["quantity"] = _num_or_none(item.get("quantity"))
        item["unit"] = str(item.get("unit") or "").strip() or None
        item["unit_price"] = _num_or_none(item.get("unit_price"))
        item["original_price"] = _num_or_none(item.get("original_price"))
        item["discount_amount"] = _num_or_none(item.get("discount_amount"))
        item["line_total"] = _num_or_none(item.get("line_total"))
        item["tax_rate"] = _num_or_none(item.get("tax_rate"))
        item["tax_code"] = str(item.get("tax_code") or "").strip() or None
        item["table_interpretation_source_row_id"] = (
            str(
                item.get("table_interpretation_source_row_id") or item.get("source_row_id") or ""
            ).strip()
            or None
        )
        item["category"] = str(item.get("category") or "item")
        item["source_line_ids"] = _list_of_str(item.get("source_line_ids"))
        try:
            item["confidence"] = max(0.0, min(1.0, float(item.get("confidence", 0.6))))
        except Exception:
            item["confidence"] = 0.6
        items.append(item)
    receipt["items"] = items

    taxes: list[dict[str, Any]] = []
    for raw in receipt.get("taxes") or []:
        if not isinstance(raw, dict):
            continue
        taxes.append(
            {
                "rate": _num_or_none(raw.get("rate")),
                "net": _num_or_none(raw.get("net")),
                "tax": _num_or_none(raw.get("tax")),
                "gross": _num_or_none(raw.get("gross")),
                "source_line_ids": _list_of_str(raw.get("source_line_ids")),
            }
        )
    receipt["taxes"] = taxes

    totals_raw = receipt.get("totals") if isinstance(receipt.get("totals"), dict) else {}
    receipt["totals"] = {
        "subtotal": _num_or_none(totals_raw.get("subtotal")),
        "tax_total": _num_or_none(totals_raw.get("tax_total")),
        "grand_total": _num_or_none(
            totals_raw.get("grand_total")
            if totals_raw.get("grand_total") is not None
            else totals_raw.get("total")
        ),
        "paid_total": _num_or_none(totals_raw.get("paid_total")),
        "change": _num_or_none(totals_raw.get("change")),
        "source_line_ids": _list_of_str(totals_raw.get("source_line_ids")),
    }

    payments: list[dict[str, Any]] = []
    for raw in receipt.get("payments") or []:
        if not isinstance(raw, dict):
            continue
        payments.append(
            {
                "method": str(raw.get("method") or raw.get("type") or "unknown")
                .strip()
                .lower()
                .replace(" ", "_"),
                "amount": _num_or_none(raw.get("amount")),
                "source_line_ids": _list_of_str(
                    raw.get("source_line_ids") or raw.get("source_row_id")
                ),
                "raw_label": raw.get("raw_label"),
                "sign_meaning": raw.get("sign_meaning"),
            }
        )
    receipt["payments"] = payments

    unresolved: list[dict[str, Any]] = []
    for raw in receipt.get("unresolved_rows") or []:
        if isinstance(raw, dict):
            unresolved.append(
                {
                    "line_id": str(raw.get("line_id") or ""),
                    "text": str(raw.get("text") or ""),
                    "reason": str(raw.get("reason") or "unresolved"),
                }
            )
    receipt["unresolved_rows"] = unresolved
    receipt["warnings"] = [str(x) for x in (receipt.get("warnings") or []) if str(x).strip()]
    try:
        receipt["overall_confidence"] = max(
            0.0, min(1.0, float(receipt.get("overall_confidence", 0.6)))
        )
    except Exception:
        receipt["overall_confidence"] = 0.6
    return receipt


def failed_receipt(error: str) -> dict[str, Any]:
    return {
        "schema_version": "v14_6_llm_receipt_1",
        "parse_status": "failed",
        "currency": "EUR",
        "merchant": {"name": None, "address": None, "tax_id": None, "source_line_ids": []},
        "date": None,
        "time": None,
        "items": [],
        "taxes": [],
        "totals": {
            "subtotal": None,
            "tax_total": None,
            "grand_total": None,
            "paid_total": None,
            "change": None,
            "source_line_ids": [],
        },
        "payments": [],
        "unresolved_rows": [],
        "warnings": [f"LLM main parser failed: {error}"],
        "overall_confidence": 0.0,
    }


def run_llm_main_parser(
    *,
    ocr_json_path: Path,
    ollama_url: str,
    model: str,
    max_lines: int = 260,
    num_ctx: int = 24384,
    num_predict: int = 8192,
    keep_alive: str | None = None,
    timeout: float = 240.0,
    json_retry_count: int = 1,
    format_json: bool = True,
    visual_evidence: dict[str, Any] | None = None,
    llm_gateway: LlmGateway | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    ocr = load_json(ocr_json_path)
    context = build_ocr_context(ocr, max_lines=max_lines)
    if visual_evidence:
        context["visual_evidence_lines"] = _visual_evidence_line_registry(visual_evidence)

    attempts: list[dict[str, Any]] = []
    receipt: dict[str, Any] | None = None
    error = None
    final_prompt = ""
    final_raw_text = ""

    # V14.6 starts with compact evidence, not the bulky V14.1 full OCR JSON prompt.
    # Retry attempts use the same evidence but stronger schema-retry wording.
    attempt_specs: list[dict[str, Any]] = [
        {
            "name": "compact_evidence_json",
            "profile": "compact_evidence",
            "format_json": format_json,
            "num_predict": max(num_predict, 6144),
        },
    ]
    for i in range(max(0, json_retry_count)):
        attempt_specs.append(
            {
                "name": f"wrong_schema_or_json_retry_{i + 1}",
                "profile": "wrong_schema_retry",
                "format_json": format_json,
                "num_predict": max(num_predict, 8192),
            }
        )
    if format_json and json_retry_count > 0:
        attempt_specs.append(
            {
                "name": "ultra_compact_no_json_grammar_last_resort",
                "profile": "ultra_compact",
                "format_json": False,
                "num_predict": max(num_predict, 8192),
            }
        )

    previous_error_for_prompt: str | None = None
    for attempt_index, spec in enumerate(attempt_specs):
        prompt = build_prompt(
            context,
            prompt_profile=spec["profile"],
            previous_error=previous_error_for_prompt,
            visual_evidence=visual_evidence,
        )
        raw_text = ""
        attempt_error = None
        attempt_started = time.perf_counter()
        try:
            generation = (
                llm_gateway.generate(
                    GenerationRequest(
                        model=model,
                        prompt=prompt,
                        operation="receipt_main_parse",
                        attempt=attempt_index + 1,
                        num_ctx=num_ctx,
                        num_predict=int(spec["num_predict"]),
                        keep_alive=keep_alive,
                        timeout_seconds=timeout,
                        format_json=bool(spec["format_json"]),
                    )
                )
                if llm_gateway is not None
                else coerce_generation_result(
                    ollama_generate(
                        ollama_url=ollama_url,
                        model=model,
                        prompt=prompt,
                        num_ctx=num_ctx,
                        num_predict=int(spec["num_predict"]),
                        keep_alive=keep_alive,
                        timeout=timeout,
                        format_json=bool(spec["format_json"]),
                    )
                )
            )
            raw_text = generation.text
            parsed = parse_json_from_llm(generation)
            parsed = normalize_schema_version_alias(parsed)
            parsed = normalize_noncritical_schema_defaults(parsed)
            validate_llm_receipt_schema_object(parsed)
            receipt = normalize_llm_receipt(parsed)
            final_prompt = prompt
            final_raw_text = raw_text
            attempts.append(
                {
                    "name": spec["name"],
                    "profile": spec["profile"],
                    "format_json": spec["format_json"],
                    "num_predict": spec["num_predict"],
                    "prompt_chars": len(prompt),
                    "raw_chars": len(raw_text),
                    "duration_seconds": round(time.perf_counter() - attempt_started, 2),
                    "status": "ok",
                }
            )
            error = None
            break
        except Exception as exc:
            attempt_error = f"{type(exc).__name__}: {exc}"
            previous_error_for_prompt = attempt_error
            final_prompt = prompt
            final_raw_text = raw_text
            error = attempt_error
            attempts.append(
                {
                    "name": spec["name"],
                    "profile": spec["profile"],
                    "format_json": spec["format_json"],
                    "num_predict": spec["num_predict"],
                    "prompt_chars": len(prompt),
                    "raw_chars": len(raw_text),
                    "duration_seconds": round(time.perf_counter() - attempt_started, 2),
                    "status": "error",
                    "error": attempt_error,
                    "raw_head": (raw_text or "")[:240],
                    "raw_tail": (raw_text or "")[-240:],
                }
            )
            continue

    if receipt is None:
        receipt = failed_receipt(error or "LLM main parser failed without returning JSON")

    duration = round(time.perf_counter() - started, 2)
    return {
        "receipt": receipt,
        "ocr_context": context,
        "prompt": final_prompt,
        "raw_output": final_raw_text,
        "error": error,
        "duration_seconds": duration,
        "model": model,
        "ollama_url": ollama_url,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "attempts": attempts,
        "visual_evidence_used": bool(
            visual_evidence and visual_evidence.get("status") in {"ok", "no_amounts_found"}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V14 LLM-main receipt parser")
    parser.add_argument("ocr_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-out", type=Path)
    parser.add_argument("--prompt-out", type=Path)
    parser.add_argument("--raw-out", type=Path)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--max-lines", type=int, default=260)
    parser.add_argument("--num-ctx", type=int, default=24384)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--json-retry-count", type=int, default=1)
    parser.add_argument(
        "--no-format-json", action="store_true", help="Do not ask Ollama to enforce JSON grammar"
    )
    args = parser.parse_args()

    result = run_llm_main_parser(
        ocr_json_path=args.ocr_json,
        ollama_url=args.ollama_url,
        model=args.model,
        max_lines=args.max_lines,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        json_retry_count=args.json_retry_count,
        format_json=not args.no_format_json,
    )
    save_json(args.out, result["receipt"])
    if args.context_out:
        save_json(args.context_out, result["ocr_context"])
    if args.prompt_out:
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_out.write_text(result["prompt"], encoding="utf-8")
    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text(result["raw_output"] or "", encoding="utf-8")
    if result.get("error"):
        print(f"LLM main parser failed without deterministic fallback: {result['error']}")
        return 2
    print(f"Wrote {args.out} with {len(result['receipt'].get('items', []))} item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
