"""Conservative field-level line-price repair from fused spatial evidence."""

from __future__ import annotations

import copy
import re
from typing import Any

from receipt_intelligence.extraction.evidence.region_price_fusion import (
    description_similarity,
    normalize_description,
)

JsonObject = dict[str, Any]

_PURCHASE_CATEGORIES = {"item", "product", "purchase_item", "purchased_product"}
_TRIGGER_ISSUES = {"ITEM_SUM_MISMATCH", "ITEMS_WITHOUT_LINE_TOTAL"}


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _close(left: Any, right: Any, tolerance: float) -> bool:
    a = _float(left)
    b = _float(right)
    return a is not None and b is not None and abs(a - b) <= tolerance


def _item_description(item: JsonObject) -> str:
    return str(
        item.get("product_description")
        or item.get("description")
        or item.get("raw_description")
        or ""
    ).strip()


def _purchase_item(item: JsonObject) -> bool:
    category = str(item.get("category") or "item").strip().lower()
    return category in _PURCHASE_CATEGORIES


def _candidate_matches(
    items: list[JsonObject], candidates: list[JsonObject]
) -> list[tuple[int, JsonObject, float]]:
    """Produce one-to-one, order-aware item/candidate matches."""

    matches: list[tuple[int, JsonObject, float]] = []
    used_items: set[int] = set()
    item_count = max(1, len(items))
    candidate_count = max(1, len(candidates))
    for candidate_index, candidate in enumerate(candidates):
        candidate_description = str(candidate.get("description") or "")
        scored: list[tuple[float, int]] = []
        for item_index, item in enumerate(items):
            if item_index in used_items or not _purchase_item(item):
                continue
            semantic = description_similarity(candidate_description, _item_description(item))
            if semantic < 0.78:
                continue
            order_distance = abs(
                item_index / item_count - candidate_index / candidate_count
            )
            score = semantic - min(0.08, order_distance * 0.08)
            scored.append((score, item_index))
        if not scored:
            continue
        scored.sort(key=lambda value: value[0], reverse=True)
        score, item_index = scored[0]
        exact = normalize_description(candidate_description) == normalize_description(
            _item_description(items[item_index])
        )
        if score < 0.84 and not exact:
            continue
        if len(scored) > 1 and not exact and score - scored[1][0] < 0.06:
            continue
        used_items.add(item_index)
        matches.append((item_index, candidate, round(score, 4)))
    return matches


def _spatial_line_index(document_map: JsonObject) -> dict[str, JsonObject]:
    return {
        str(row.get("line_id")): row
        for row in document_map.get("rows") or []
        if isinstance(row, dict) and row.get("line_id")
    }


def _row_amounts(row: JsonObject) -> list[float]:
    amounts: list[float] = []
    for cell in row.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        for candidate in cell.get("amount_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            value = _float(candidate.get("value"))
            if value is not None:
                amounts.append(round(value, 2))
    return amounts


def _right_side(row: JsonObject) -> bool:
    box = row.get("bbox") or {}
    x_center = (_float(box.get("x"), 0.0) or 0.0) + (_float(box.get("w"), 0.0) or 0.0) / 2.0
    return x_center >= 0.65


def _remove_conflicting_price_sources(
    source_ids: list[str],
    *,
    old_value: float | None,
    new_value: float,
    line_index: dict[str, JsonObject],
    tolerance: float,
) -> list[str]:
    kept: list[str] = []
    for source_id in source_ids:
        row = line_index.get(source_id)
        if not isinstance(row, dict) or not _right_side(row):
            kept.append(source_id)
            continue
        amounts = _row_amounts(row)
        row_confidence = _float(row.get("confidence"), 1.0) or 0.0
        text = str(row.get("text") or "")
        conflicts = any(not _close(amount, new_value, tolerance) for amount in amounts)
        represented_old = old_value is not None and any(
            _close(amount, old_value, tolerance) for amount in amounts
        )
        damaged_numeric = not amounts and row_confidence < 0.75 and bool(re.search(r"\d", text))
        if conflicts and represented_old or damaged_numeric:
            continue
        kept.append(source_id)
    return kept


def repair_receipt_line_prices(
    receipt: JsonObject,
    validation_report: JsonObject,
    spatial_document_map: JsonObject | None,
    *,
    tolerance: float = 0.03,
    minimum_confidence: float = 0.80,
) -> tuple[JsonObject, list[JsonObject]]:
    """Patch only item money fields supported by exact region crop evidence.

    The function does not add, remove, reorder, or rename items and never changes
    totals, taxes, payments, or merchant data. Candidate selection remains
    validation-gated by the caller.
    """

    if not isinstance(spatial_document_map, dict):
        return copy.deepcopy(receipt), []
    issue_codes = {
        str(issue.get("code"))
        for issue in validation_report.get("issues") or []
        if isinstance(issue, dict)
    }
    if not issue_codes.intersection(_TRIGGER_ISSUES):
        return copy.deepcopy(receipt), []

    candidates = [
        candidate
        for candidate in spatial_document_map.get("region_item_price_candidates") or []
        if isinstance(candidate, dict)
        and _float(candidate.get("line_total")) is not None
        and (_float(candidate.get("layout_confidence"), 0.0) or 0.0) >= minimum_confidence
        and len(candidate.get("source_line_ids") or []) >= 2
    ]
    if not candidates:
        return copy.deepcopy(receipt), []

    result = copy.deepcopy(receipt)
    items = [item for item in result.get("items") or [] if isinstance(item, dict)]
    line_index = _spatial_line_index(spatial_document_map)
    actions: list[JsonObject] = []

    for item_index, candidate, match_score in _candidate_matches(items, candidates):
        item = items[item_index]
        new_total = round(float(candidate["line_total"]), 2)
        old_total = _float(item.get("line_total"))
        exact_description = normalize_description(candidate.get("description")) == normalize_description(
            _item_description(item)
        )
        strong_match = exact_description or match_score >= 0.92
        if not strong_match:
            continue

        candidate_quantity = _float(candidate.get("quantity"))
        candidate_unit_price = _float(candidate.get("unit_price"))
        candidate_unit = candidate.get("unit")
        quantity_supported = (
            candidate_quantity is not None
            and candidate_unit_price is not None
            and _close(
                candidate_quantity * candidate_unit_price,
                new_total,
                max(tolerance, 0.05),
            )
        )
        line_total_changed = not _close(old_total, new_total, tolerance)
        quantity_changed = quantity_supported and not _close(
            item.get("quantity"), candidate_quantity, 0.0005
        )
        unit_price_changed = quantity_supported and not _close(
            item.get("unit_price"), candidate_unit_price, tolerance
        )
        unit_changed = bool(
            quantity_supported
            and candidate_unit
            and str(item.get("unit") or "").strip().casefold()
            != str(candidate_unit).strip().casefold()
        )
        changed_fields = [
            field
            for field, changed in (
                ("line_total", line_total_changed),
                ("quantity", quantity_changed),
                ("unit_price", unit_price_changed),
                ("unit", unit_changed),
            )
            if changed
        ]
        if not changed_fields:
            continue

        source_ids = [str(value) for value in item.get("source_line_ids") or [] if value]
        if line_total_changed:
            source_ids = _remove_conflicting_price_sources(
                source_ids,
                old_value=old_total,
                new_value=new_total,
                line_index=line_index,
                tolerance=tolerance,
            )
        source_ids.extend(str(value) for value in candidate.get("source_line_ids") or [] if value)
        quantity_source_ids = [
            str(value) for value in candidate.get("quantity_source_line_ids") or [] if value
        ]
        if quantity_supported:
            source_ids.extend(quantity_source_ids)

        if line_total_changed:
            item["line_total"] = new_total
        if quantity_supported:
            if quantity_changed:
                item["quantity"] = round(float(candidate_quantity), 3)
            if unit_price_changed:
                item["unit_price"] = round(float(candidate_unit_price), 3)
            if unit_changed:
                item["unit"] = candidate_unit
        item["source_line_ids"] = list(dict.fromkeys(source_ids))
        item["confidence"] = max(
            _float(item.get("confidence"), 0.0) or 0.0,
            min(0.99, _float(candidate.get("layout_confidence"), 0.0) or 0.0),
        )
        actions.append(
            {
                "action": "patch_item_money_fields_from_region_spatial_evidence",
                "item_index": item_index,
                "description": _item_description(item),
                "changed_fields": changed_fields,
                "old_line_total": old_total,
                "new_line_total": new_total if line_total_changed else old_total,
                "candidate_id": candidate.get("candidate_id"),
                "description_match_score": match_score,
                "evidence_source": candidate.get("evidence_source"),
                "source_line_ids": candidate.get("source_line_ids") or [],
                "quantity_supported": quantity_supported,
            }
        )

    result["items"] = items
    return result, actions


__all__ = ["repair_receipt_line_prices"]
