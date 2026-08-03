from __future__ import annotations

import difflib
import re
import unicodedata
from decimal import Decimal
from typing import Any, Sequence

from ..evidence import money_float, parse_decimal_literal, parse_rows

_MAX_PATCHES = 8
_MAX_INSERTIONS = 4
_MAX_REPLACEMENTS = 8
_MIN_NAME_SCORE = 0.92
_MIN_NAME_MARGIN = 0.08
_PER_UNIT_MARKER = re.compile(
    r"(?:/|\bpro\b|\bper\b)\s*(?:kg|g|l|ml|cl|m|cm|stk|st|stück|piece)\b",
    re.IGNORECASE,
)
_MULTIPLICATION = re.compile(r"\b[0-9]+(?:[.,][0-9]+)?\s*[x×]\s*", re.IGNORECASE)
_MONEY_TOKEN = re.compile(
    r"(?<![0-9])[-+]?(?:[0-9]{1,3}(?:[ .][0-9]{3})*|[0-9]+)[.,][0-9]{2}(?![0-9])"
)


def validate_item_sum_evidence(answer: Any, transcription: str) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        rows, source_warnings = parse_rows(transcription)
    except Exception as exc:
        return {
            "status": "invalid",
            "errors": [{"code": "SOURCE_ROWS_UNAVAILABLE", "message": str(exc)}],
            "warnings": [],
            "metrics": {"item_block_count": 0, "unresolved_group_count": 0},
        }
    warnings.extend(
        {"code": "TRANSCRIPTION_WARNING", "message": value}
        for value in source_warnings
    )
    source = {row["row_id"]: row["text"] for row in rows}
    order = {row["row_id"]: index for index, row in enumerate(rows)}

    if not isinstance(answer, dict):
        return {
            "status": "invalid",
            "errors": [{"code": "ANSWER_NOT_OBJECT"}],
            "warnings": warnings,
            "metrics": {"item_block_count": 0, "unresolved_group_count": 0},
        }
    if set(answer) != {"item_blocks", "unresolved_candidate_rows"}:
        errors.append({"code": "TOP_LEVEL_FIELDS_INVALID"})
    blocks = answer.get("item_blocks")
    unresolved = answer.get("unresolved_candidate_rows")
    if not isinstance(blocks, list):
        errors.append({"code": "ITEM_BLOCKS_NOT_ARRAY"})
        blocks = []
    if not isinstance(unresolved, list):
        errors.append({"code": "UNRESOLVED_ROWS_NOT_ARRAY"})
        unresolved = []
    if len(blocks) > 200:
        errors.append({"code": "TOO_MANY_ITEM_BLOCKS"})
    if len(unresolved) > 100:
        errors.append({"code": "TOO_MANY_UNRESOLVED_GROUPS"})

    used_rows: set[str] = set()

    def validate_rows(value: Any, location: str) -> list[str]:
        if not isinstance(value, list) or not value or len(value) > 16:
            errors.append({"code": "INVALID_SOURCE_ROWS", "location": location})
            return []
        valid: list[str] = []
        indices: list[int] = []
        for row_id in value:
            if not isinstance(row_id, str) or row_id not in source:
                errors.append(
                    {"code": "UNKNOWN_SOURCE_ROW", "location": location, "value": row_id}
                )
                continue
            if row_id in used_rows:
                errors.append(
                    {"code": "SOURCE_ROW_REUSED", "location": location, "value": row_id}
                )
            used_rows.add(row_id)
            valid.append(row_id)
            indices.append(order[row_id])
        if len(valid) != len(set(valid)):
            errors.append({"code": "DUPLICATE_SOURCE_ROW", "location": location})
        if indices != sorted(indices):
            errors.append({"code": "SOURCE_ROWS_OUT_OF_ORDER", "location": location})
        return valid

    for index, block in enumerate(blocks):
        location = f"item_blocks[{index}]"
        if not isinstance(block, dict):
            errors.append({"code": "ITEM_BLOCK_NOT_OBJECT", "location": location})
            continue
        allowed_fields = frozenset({"source_rows", "name", "line_amount", "unit_price"})
        extended_fields = allowed_fields | {"original_price", "discount_amount"}
        if frozenset(block) not in {allowed_fields, extended_fields}:
            errors.append({"code": "ITEM_BLOCK_FIELDS_INVALID", "location": location})
        row_ids = validate_rows(block.get("source_rows"), f"{location}.source_rows")
        joined = "\n".join(source[row_id] for row_id in row_ids)
        for field_name in ("name", "line_amount"):
            value = block.get(field_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    {"code": f"INVALID_{field_name.upper()}", "location": f"{location}.{field_name}"}
                )
            elif value not in joined:
                errors.append(
                    {"code": "VALUE_NOT_LITERAL_IN_SOURCE_ROWS", "location": f"{location}.{field_name}", "value": value}
                )
        for field_name in ("unit_price", "original_price", "discount_amount"):
            value = block.get(field_name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    {
                        "code": f"INVALID_{field_name.upper()}",
                        "location": f"{location}.{field_name}",
                    }
                )
            elif value not in joined:
                errors.append(
                    {
                        "code": "VALUE_NOT_LITERAL_IN_SOURCE_ROWS",
                        "location": f"{location}.{field_name}",
                        "value": value,
                    }
                )

    for index, group in enumerate(unresolved):
        location = f"unresolved_candidate_rows[{index}]"
        if not isinstance(group, dict) or set(group) != {"source_rows"}:
            errors.append({"code": "UNRESOLVED_GROUP_INVALID", "location": location})
            continue
        validate_rows(group.get("source_rows"), f"{location}.source_rows")

    return {
        "status": "invalid" if errors else "valid",
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "item_block_count": len(blocks),
            "unresolved_group_count": len(unresolved),
            "referenced_source_row_count": len(used_rows),
        },
    }


def _parse_line_amount(value: Any) -> Decimal | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text or _MULTIPLICATION.search(text) or _PER_UNIT_MARKER.search(text):
        return None
    if len(_MONEY_TOKEN.findall(text)) != 1:
        return None
    amount = parse_decimal_literal(text)
    if amount is None or amount < 0:
        return None
    return amount


def _parse_optional_amount(value: Any, *, positive_magnitude: bool = False) -> Decimal | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text or len(_MONEY_TOKEN.findall(text)) != 1:
        return None
    amount = parse_decimal_literal(text)
    if amount is None:
        return None
    if positive_magnitude:
        amount = abs(amount)
    elif amount < 0:
        return None
    return amount


def _normalize_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    without_money = _MONEY_TOKEN.sub(" ", without_marks)
    tokens = re.findall(r"[a-z0-9]+", without_money)
    return " ".join(
        token
        for token in tokens
        if not re.fullmatch(r"[0-9]+(?:kg|g|l|ml|cl|dl|m|cm|mm|stk|st|x)?", token)
    )


def _money(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except Exception:
        return None


def _money_close(left: Decimal, right: Decimal) -> bool:
    return abs(left - right) <= Decimal("0.01")


def _match_blocks(
    source_blocks: Sequence[dict[str, Any]],
    current_items: Sequence[dict[str, Any]],
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    current_names = [
        _normalize_name(item.get("name")) if isinstance(item, dict) else ""
        for item in current_items
    ]
    available = set(range(len(current_items)))
    matches: dict[int, int] = {}
    diagnostics: list[dict[str, Any]] = []
    last_index = -1
    for source_index, block in enumerate(source_blocks):
        source_name = _normalize_name(block.get("name"))
        if not source_name:
            continue
        candidates = sorted(index for index in available if index > last_index)
        exact = [index for index in candidates if current_names[index] == source_name]
        if exact:
            selected, score, method = exact[0], 1.0, "exact"
        else:
            scored = sorted(
                (
                    difflib.SequenceMatcher(None, source_name, current_names[index]).ratio(),
                    index,
                )
                for index in candidates
                if current_names[index]
            )
            scored.reverse()
            if not scored:
                continue
            best_score, selected = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            if best_score < _MIN_NAME_SCORE or best_score - second_score < _MIN_NAME_MARGIN:
                diagnostics.append(
                    {
                        "source_index": source_index,
                        "source_name": block.get("name"),
                        "status": "unmatched_or_ambiguous",
                        "best_score": round(best_score, 4),
                        "second_score": round(second_score, 4),
                    }
                )
                continue
            score, method = best_score, "fuzzy_unique"
        matches[source_index] = selected
        available.remove(selected)
        last_index = selected
        diagnostics.append(
            {
                "source_index": source_index,
                "source_name": block.get("name"),
                "current_index": selected,
                "current_name": current_items[selected].get("name"),
                "method": method,
                "score": round(score, 4),
            }
        )
    return matches, diagnostics


def build_item_sum_patch(
    source_answer: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_items = receipt.get("items")
    if not isinstance(current_items, list) or not current_items:
        return {"patches": []}, {"status": "abstained", "reason": "current_items_missing_or_empty"}

    raw_blocks = source_answer.get("item_blocks")
    raw_blocks = raw_blocks if isinstance(raw_blocks, list) else []
    source_blocks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, block in enumerate(raw_blocks):
        if not isinstance(block, dict):
            continue
        name = block.get("name")
        amount = _parse_line_amount(block.get("line_amount"))
        if not isinstance(name, str) or not name.strip():
            rejected.append({"index": index, "reason": "missing_name"})
            continue
        source_blocks.append(
            {
                **block,
                "_amount": amount,
                "_original_price": _parse_optional_amount(block.get("original_price")),
                "_discount_amount": _parse_optional_amount(
                    block.get("discount_amount"),
                    positive_magnitude=True,
                ),
            }
        )
        if amount is None:
            rejected.append(
                {
                    "index": index,
                    "name": name,
                    "reason": "line_amount_not_explicit_standalone_money",
                    "value": block.get("line_amount"),
                }
            )

    if not source_blocks:
        return {"patches": []}, {"status": "abstained", "reason": "no_usable_source_item_blocks", "rejected_blocks": rejected}

    matches, match_diagnostics = _match_blocks(source_blocks, current_items)
    matched_count = len(matches)
    minimum_matches = min(3, len(current_items))
    coverage = matched_count / max(1, len(current_items))
    if matched_count < minimum_matches or coverage < 0.60:
        return {"patches": []}, {
            "status": "abstained",
            "reason": "insufficient_match_coverage",
            "matched_count": matched_count,
            "current_item_count": len(current_items),
            "coverage": round(coverage, 4),
            "matches": match_diagnostics,
            "rejected_blocks": rejected,
        }

    replacements: list[dict[str, Any]] = []
    unmatched: list[int] = []
    for source_index, block in enumerate(source_blocks):
        source_amount = block.get("_amount")
        if source_amount is None:
            continue
        current_index = matches.get(source_index)
        if current_index is None:
            unmatched.append(source_index)
            continue
        source_rows = ",".join(block.get("source_rows") or [])
        candidates = (
            ("final_price", source_amount, block.get("line_amount")),
            ("original_price", block.get("_original_price"), block.get("original_price")),
            ("discount_amount", block.get("_discount_amount"), block.get("discount_amount")),
        )
        for field_name, source_value, printed_value in candidates:
            if source_value is None or field_name not in current_items[current_index]:
                continue
            current_value = _money(current_items[current_index].get(field_name))
            if current_value is not None and _money_close(current_value, source_value):
                continue
            replacements.append(
                {
                    "op": "replace_value",
                    "reason": (
                        f"Source rows {source_rows} print item {block['name']!r} "
                        f"with {field_name} evidence {printed_value!r}."
                    )[:240],
                    "path": f"/items/{current_index}/{field_name}",
                    "value": money_float(source_value),
                }
            )

    if len(replacements) > _MAX_REPLACEMENTS:
        return {"patches": []}, {"status": "abstained", "reason": "too_many_item_field_replacements", "replacement_count": len(replacements)}
    if len(unmatched) > _MAX_INSERTIONS:
        return {"patches": []}, {"status": "abstained", "reason": "too_many_missing_source_items", "insertion_count": len(unmatched)}

    insertions: list[dict[str, Any]] = []
    positions: list[int | None] = list(range(len(current_items)))
    cursor = 0
    unmatched_set = set(unmatched)
    for source_index, block in enumerate(source_blocks):
        current_index = matches.get(source_index)
        if current_index is not None:
            try:
                cursor = positions.index(current_index) + 1
            except ValueError:
                return {"patches": []}, {"status": "abstained", "reason": "matched_item_position_lost"}
            continue
        if source_index not in unmatched_set or block.get("_amount") is None:
            continue
        item_value = {
            "name": block["name"].strip(),
            "final_price": money_float(block["_amount"]),
            "quantity": None,
            "unit": None,
            "discount_amount": (
                money_float(block["_discount_amount"])
                if block.get("_discount_amount") is not None
                else None
            ),
            "original_price": (
                money_float(block["_original_price"])
                if block.get("_original_price") is not None
                else None
            ),
        }
        insertions.append(
            {
                "op": "insert_array_element",
                "reason": (
                    f"Source rows {','.join(block.get('source_rows') or [])} contain "
                    f"independent item {block['name']!r} with printed line amount "
                    f"{block.get('line_amount')!r}."
                )[:240],
                "path": "/items",
                "index": cursor,
                "value": item_value,
            }
        )
        positions.insert(cursor, None)
        cursor += 1

    patches = replacements + insertions
    if not patches:
        return {"patches": []}, {
            "status": "abstained",
            "reason": "source_inventory_matches_current_items",
            "matches": match_diagnostics,
            "rejected_blocks": rejected,
        }
    if len(patches) > _MAX_PATCHES:
        return {"patches": []}, {"status": "abstained", "reason": "patch_limit_exceeded", "patch_count": len(patches)}
    return {"patches": patches}, {
        "status": "patch_built",
        "source_item_block_count": len(source_blocks),
        "matched_count": matched_count,
        "match_coverage": round(coverage, 4),
        "replacement_count": len(replacements),
        "insertion_count": len(insertions),
        "matches": match_diagnostics,
        "rejected_blocks": rejected,
    }
