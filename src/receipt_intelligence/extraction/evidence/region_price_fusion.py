"""Fuse high-confidence region crop OCR prices into spatial receipt evidence.

The full-image OCR remains the canonical page coordinate source. Region crop OCR
is used as supplemental evidence because it often reads small right-column
prices more accurately. This module never creates receipt semantics and never
reconstructs an item list. It only exposes auditable product/price candidates
with source IDs, coordinates, confidence, and optional quantity support.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from statistics import median
from typing import Any

JsonObject = dict[str, Any]

_PRODUCT_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{2,}")
_QUANTITY_RE = re.compile(
    r"(?P<quantity>\d+(?:[,.]\d+)?)\s*"
    r"(?P<unit>STK|STÜCK|STUECK|PCS?|KG|G|L|ML)?\s*(?:x|×|\*)",
    re.IGNORECASE,
)
_UNIT_PRICE_RE = re.compile(
    r"(?P<amount>\d+(?:[,.]\d+)?)\s*(?:EUR|EURO|€)?\s*/\s*"
    r"(?P<unit>STK|STÜCK|STUECK|PCS?|KG|G|L|ML)",
    re.IGNORECASE,
)
_TAX_CODE_RE = re.compile(r"\d+[,.]\d{2}\s*([A-Z])\b", re.IGNORECASE)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _money(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = str(value).strip().replace("−", "-")
    if not text:
        return None
    negative = text.startswith("-") or text.endswith("-")
    token = re.sub(r"[^0-9,.]", "", text)
    if not token:
        return None
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        token = token.replace(".", "").replace(",", ".")
    try:
        parsed = round(float(token), 2)
    except ValueError:
        return None
    return -abs(parsed) if negative else parsed


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().replace(",", ".")
    try:
        return round(float(text), 3)
    except (TypeError, ValueError):
        return None


def normalize_description(value: Any) -> str:
    """Return a punctuation-insensitive description key for evidence matching."""

    text = unicodedata.normalize("NFKD", str(value or "")).upper()
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def description_similarity(left: Any, right: Any) -> float:
    """Score two printed product descriptions without merchant-specific rules."""

    a = normalize_description(left)
    b = normalize_description(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if min(len(a), len(b)) >= 8 and (a.startswith(b) or b.startswith(a)):
        return 0.97
    ratio = SequenceMatcher(None, a, b).ratio()
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    overlap = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    containment = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
    return round(max(ratio, ratio * 0.65 + overlap * 0.35, containment * 0.88), 4)


def _bbox_from_region_line(line: JsonObject, page_width: int, page_height: int) -> JsonObject:
    width = max(1, int(page_width))
    height = max(1, int(page_height))
    x0 = max(0.0, _float(line.get("xmin")))
    y0 = max(0.0, _float(line.get("ymin")))
    x1 = max(x0, _float(line.get("xmax")))
    y1 = max(y0, _float(line.get("ymax")))
    return {
        "x": round(min(1.0, x0 / width), 5),
        "y": round(min(1.0, y0 / height), 5),
        "w": round(min(1.0, max(0.0, x1 - x0) / width), 5),
        "h": round(min(1.0, max(0.0, y1 - y0) / height), 5),
    }


def _line_center_y(line: JsonObject) -> float:
    if line.get("y_center") is not None:
        return _float(line.get("y_center"))
    return (_float(line.get("ymin")) + _float(line.get("ymax"))) / 2.0


def _line_center_x(line: JsonObject) -> float:
    if line.get("x_center") is not None:
        return _float(line.get("x_center"))
    return (_float(line.get("xmin")) + _float(line.get("xmax"))) / 2.0


def _line_height(line: JsonObject) -> float:
    return max(1.0, _float(line.get("ymax")) - _float(line.get("ymin")))


def _region_line_index(region_reocr: JsonObject) -> tuple[dict[str, JsonObject], list[JsonObject]]:
    index: dict[str, JsonObject] = {}
    ordered: list[JsonObject] = []
    for region in region_reocr.get("regions") or []:
        if not isinstance(region, dict):
            continue
        region_id = str(region.get("region_id") or "")
        for raw in region.get("lines") or []:
            if not isinstance(raw, dict):
                continue
            line = dict(raw)
            line["region_id"] = region_id
            line_id = str(line.get("id") or "")
            if not line_id:
                continue
            index[line_id] = line
            ordered.append(line)
    ordered.sort(key=lambda row: (_line_center_y(row), _line_center_x(row)))
    return index, ordered


def _candidate_tax_code(amount_line: JsonObject | None) -> str | None:
    if not isinstance(amount_line, dict):
        return None
    match = _TAX_CODE_RE.search(str(amount_line.get("text") or ""))
    return match.group(1).upper() if match else None


def _parse_quantity_support(
    *,
    row_id: str,
    block: JsonObject,
    line_index: dict[str, JsonObject],
    page_width: int,
) -> JsonObject:
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    source_ids: list[str] = []

    links = [
        link
        for link in block.get("quantity_note_links") or []
        if isinstance(link, dict) and str(link.get("linked_item_row_id") or "") == row_id
    ]
    for link in links:
        line_id = str(link.get("quantity_row_id") or "")
        text = str(link.get("quantity_text") or "")
        if line_id:
            source_ids.append(line_id)
        quantity_match = _QUANTITY_RE.search(text)
        if quantity_match:
            quantity = _number(quantity_match.group("quantity"))
            raw_unit = quantity_match.group("unit")
            unit = raw_unit if raw_unit else unit
        unit_price_match = _UNIT_PRICE_RE.search(text)
        if unit_price_match:
            unit_price = _money(unit_price_match.group("amount"))
            raw_unit = unit_price_match.group("unit")
            unit = raw_unit if raw_unit else unit

    if quantity is not None and unit_price is None:
        quantity_lines = [line_index[line_id] for line_id in source_ids if line_id in line_index]
        for quantity_line in quantity_lines:
            qy = _line_center_y(quantity_line)
            qh = _line_height(quantity_line)
            candidates: list[tuple[float, JsonObject, float]] = []
            for line in line_index.values():
                amounts = line.get("amounts") or []
                if not amounts:
                    continue
                x_center = _line_center_x(line)
                # Unit prices normally occupy a middle column. The rightmost
                # column is reserved for line totals.
                if x_center >= max(1, page_width) * 0.72:
                    continue
                dy = abs(_line_center_y(line) - qy)
                if dy > max(18.0, qh * 0.85):
                    continue
                amount = _money((amounts[-1] or {}).get("value"))
                if amount is None:
                    continue
                candidates.append((dy, line, amount))
            if candidates:
                candidates.sort(key=lambda value: value[0])
                _, matched_line, matched_amount = candidates[0]
                unit_price = matched_amount
                matched_id = str(matched_line.get("id") or "")
                if matched_id:
                    source_ids.append(matched_id)
                break

    normalized_unit = unit.upper() if unit else None
    if normalized_unit in {"STÜCK", "STUECK", "PC", "PCS"}:
        normalized_unit = "Stk"
    elif normalized_unit:
        normalized_unit = normalized_unit.lower() if normalized_unit != "STK" else "Stk"

    return {
        "quantity": quantity,
        "unit": normalized_unit,
        "unit_price": unit_price,
        "source_line_ids": list(dict.fromkeys(source_ids)),
    }


def _preferred_candidates(
    region_reocr: JsonObject,
    *,
    line_index: dict[str, JsonObject],
    page_width: int,
    page_height: int,
) -> list[JsonObject]:
    candidates: list[JsonObject] = []
    for block_index, block in enumerate(region_reocr.get("preferred_item_blocks") or []):
        if not isinstance(block, dict):
            continue
        region_id = str(block.get("region_id") or f"region_{block_index:02d}")
        for _row_index, row in enumerate(block.get("rows") or []):
            if not isinstance(row, dict):
                continue
            amount = _money(row.get("amount"))
            description = str(row.get("description_candidate") or row.get("text") or "").strip()
            if amount is None or not description:
                continue
            row_id = str(row.get("row_id") or "")
            product_line = line_index.get(row_id)
            source_ids = [str(value) for value in row.get("source_line_ids") or [] if value]
            amount_line = next(
                (
                    line_index[source_id]
                    for source_id in source_ids
                    if source_id in line_index and (line_index[source_id].get("amounts") or [])
                ),
                None,
            )
            quantity = _parse_quantity_support(
                row_id=row_id,
                block=block,
                line_index=line_index,
                page_width=page_width,
            )
            if product_line is not None:
                bbox = _bbox_from_region_line(product_line, page_width, page_height)
                y_center = round(_line_center_y(product_line) / max(1, page_height), 5)
            else:
                bbox = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
                y_center = 0.0
            candidate = {
                "candidate_id": f"region_price_{len(candidates):03d}",
                "region_id": region_id,
                "description": description,
                "description_key": normalize_description(description),
                "line_total": amount,
                "amount_raw": row.get("amount_raw"),
                "tax_code": _candidate_tax_code(amount_line),
                "layout_confidence": round(_float(row.get("layout_confidence")), 4),
                "evidence_source": str(row.get("evidence_source") or "preferred_region_row"),
                "source_line_ids": list(dict.fromkeys([row_id, *source_ids])),
                "bbox": bbox,
                "y_center": y_center,
                "quantity": quantity.get("quantity"),
                "unit": quantity.get("unit"),
                "unit_price": quantity.get("unit_price"),
                "quantity_source_line_ids": quantity.get("source_line_ids") or [],
            }
            candidates.append(candidate)
    return candidates


def _fallback_same_band_candidates(
    *,
    lines: list[JsonObject],
    existing: list[JsonObject],
    page_width: int,
    page_height: int,
    eligible_product_ids: set[str] | None = None,
) -> list[JsonObject]:
    """Recover clean right-column prices omitted by the conservative block builder.

    A clean amount wins only when it is closer to the product baseline than any
    damaged right-column token. This prevents a missing price from stealing the
    next product's amount while allowing cases such as ``3,35`` above
    ``LUXUS TOILET. PAP`` to be recovered.
    """

    if not lines:
        return []
    heights = [_line_height(line) for line in lines]
    median_height = median(heights) if heights else 20.0
    products = [
        line
        for line in lines
        if str(line.get("role_hint") or "") == "product_or_item_text"
        and _PRODUCT_WORD_RE.search(str(line.get("text") or ""))
        and (
            not eligible_product_ids
            or str(line.get("id") or "") in eligible_product_ids
        )
    ]
    clean_amounts = [
        line
        for line in lines
        if line.get("amounts") and _line_center_x(line) >= max(1, page_width) * 0.68
    ]
    damaged_amounts = [
        line
        for line in lines
        if line.get("damaged_amount_candidate")
        and _line_center_x(line) >= max(1, page_width) * 0.68
    ]
    existing_keys = {str(candidate.get("description_key") or "") for candidate in existing}
    used_amount_ids = {
        source_id
        for candidate in existing
        for source_id in candidate.get("source_line_ids") or []
        if source_id
    }
    recovered: list[JsonObject] = []
    for product in products:
        description = str(product.get("text") or "").strip()
        description_key = normalize_description(description)
        if not description_key or description_key in existing_keys:
            continue
        py = _line_center_y(product)
        px = _line_center_x(product)
        y_tolerance = max(20.0, median_height * 0.72)
        clean: list[tuple[float, float, JsonObject]] = []
        for amount_line in clean_amounts:
            amount_id = str(amount_line.get("id") or "")
            if amount_id in used_amount_ids or _line_center_x(amount_line) <= px:
                continue
            dy = abs(_line_center_y(amount_line) - py)
            if dy <= y_tolerance:
                clean.append((dy, -_line_center_x(amount_line), amount_line))
        if not clean:
            continue
        clean.sort(key=lambda value: (value[0], value[1]))
        clean_dy, _, amount_line = clean[0]
        damaged_dy = min(
            (
                abs(_line_center_y(damaged) - py)
                for damaged in damaged_amounts
                if _line_center_x(damaged) > px
                and abs(_line_center_y(damaged) - py) <= y_tolerance
            ),
            default=None,
        )
        if damaged_dy is not None and damaged_dy < clean_dy:
            continue
        amount_record = (amount_line.get("amounts") or [None])[-1]
        amount = _money(amount_record.get("value") if isinstance(amount_record, dict) else None)
        if amount is None:
            continue
        amount_id = str(amount_line.get("id") or "")
        product_id = str(product.get("id") or "")
        used_amount_ids.add(amount_id)
        existing_keys.add(description_key)
        recovered.append(
            {
                "candidate_id": f"region_price_fallback_{len(recovered):03d}",
                "region_id": str(product.get("region_id") or ""),
                "description": description,
                "description_key": description_key,
                "line_total": amount,
                "amount_raw": amount_record.get("raw") if isinstance(amount_record, dict) else None,
                "tax_code": _candidate_tax_code(amount_line),
                "layout_confidence": round(
                    min(_float(product.get("confidence")), _float(amount_line.get("confidence"))),
                    4,
                ),
                "evidence_source": "region_clean_right_amount_closer_than_damaged_token",
                "source_line_ids": [product_id, amount_id],
                "bbox": _bbox_from_region_line(product, page_width, page_height),
                "y_center": round(py / max(1, page_height), 5),
                "quantity": None,
                "unit": None,
                "unit_price": None,
                "quantity_source_line_ids": [],
            }
        )
    return recovered


def _spatial_row_amounts(row: JsonObject) -> list[float]:
    values: list[float] = []
    for cell in row.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        for candidate in cell.get("amount_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            amount = _money(candidate.get("value"))
            if amount is not None:
                values.append(amount)
    return values


def _match_candidates_to_spatial_rows(
    candidates: list[JsonObject], rows: list[JsonObject]
) -> list[JsonObject]:
    product_rows = [
        row
        for row in rows
        if isinstance(row, dict) and _PRODUCT_WORD_RE.search(str(row.get("text") or ""))
    ]
    for candidate in candidates:
        scored: list[tuple[float, float, JsonObject]] = []
        for row in product_rows:
            semantic_score = description_similarity(candidate.get("description"), row.get("text"))
            if semantic_score < 0.65:
                continue
            row_box = row.get("bbox") or {}
            row_y = _float(row_box.get("y")) + _float(row_box.get("h")) / 2.0
            dy = abs(row_y - _float(candidate.get("y_center")))
            score = semantic_score - min(0.18, dy * 1.5)
            scored.append((score, -dy, row))
        if not scored:
            continue
        scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
        score, _, row = scored[0]
        if score < 0.72:
            continue
        candidate["matched_spatial_row_id"] = row.get("row_id")
        candidate["matched_line_id"] = row.get("line_id")
        candidate["matched_geometric_group_id"] = row.get("geometric_group_id")
        candidate["description_match_score"] = round(score, 4)
        candidate["primary_amount_candidates"] = _spatial_row_amounts(row)
    return candidates


def build_region_item_price_candidates(
    visual_evidence: JsonObject | None,
    *,
    spatial_rows: list[JsonObject] | None = None,
    page_width: int = 0,
    page_height: int = 0,
) -> list[JsonObject]:
    """Return supplemental region crop item-price candidates for the main parser."""

    if not isinstance(visual_evidence, dict):
        return []
    region_reocr = visual_evidence.get("region_reocr")
    if not isinstance(region_reocr, dict) or region_reocr.get("status") != "ok":
        return []
    line_index, lines = _region_line_index(region_reocr)
    candidates = _preferred_candidates(
        region_reocr,
        line_index=line_index,
        page_width=page_width,
        page_height=page_height,
    )
    eligible_product_ids = {
        str(row.get("row_id"))
        for block in region_reocr.get("preferred_item_blocks") or []
        if isinstance(block, dict)
        for row in block.get("unmatched_product_rows") or []
        if isinstance(row, dict) and row.get("row_id")
    }
    candidates.extend(
        _fallback_same_band_candidates(
            lines=lines,
            existing=candidates,
            page_width=page_width,
            page_height=page_height,
            eligible_product_ids=eligible_product_ids or None,
        )
    )
    candidates.sort(key=lambda candidate: (_float(candidate.get("y_center")), candidate["candidate_id"]))
    for index, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"region_price_{index:03d}"
    return _match_candidates_to_spatial_rows(candidates, spatial_rows or [])


def region_price_candidates_to_prompt_text(candidates: list[JsonObject], *, limit: int = 120) -> str:
    if not candidates:
        return "none"
    lines: list[str] = []
    for candidate in candidates[:limit]:
        primary = candidate.get("primary_amount_candidates") or []
        quantity_bits = []
        if candidate.get("quantity") is not None:
            quantity_bits.append(f"quantity={candidate.get('quantity')}")
        if candidate.get("unit"):
            quantity_bits.append(f"unit={candidate.get('unit')}")
        if candidate.get("unit_price") is not None:
            quantity_bits.append(f"unit_price={candidate.get('unit_price')}")
        quantity_suffix = f" {' '.join(quantity_bits)}" if quantity_bits else ""
        lines.append(
            "["
            f"{candidate.get('candidate_id')} match={candidate.get('matched_spatial_row_id')} "
            f"score={_float(candidate.get('description_match_score')):.3f} "
            f"confidence={_float(candidate.get('layout_confidence')):.3f}"
            "] "
            f"description={candidate.get('description')!r} "
            f"region_line_total={candidate.get('line_total')} "
            f"primary_amounts={primary} "
            f"source_line_ids={candidate.get('source_line_ids') or []}"
            f"{quantity_suffix}"
        )
    return "\n".join(lines)


__all__ = [
    "build_region_item_price_candidates",
    "description_similarity",
    "normalize_description",
    "region_price_candidates_to_prompt_text",
]
