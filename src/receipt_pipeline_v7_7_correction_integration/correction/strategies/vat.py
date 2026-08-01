from __future__ import annotations

from decimal import Decimal
from typing import Any

from ..evidence import money_float, parse_decimal_literal, row_map

_ROLES = {"vat_amount", "gross_amount", "rate_percent", "net_amount"}


def validate_vat_evidence(answer: Any, transcription: str) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        source, source_warnings = row_map(transcription)
    except Exception as exc:
        return {
            "status": "invalid",
            "errors": [{"code": "SOURCE_ROWS_UNAVAILABLE", "message": str(exc)}],
            "warnings": [],
            "metrics": {"block_count": 0, "unresolved_group_count": 0},
        }
    warnings.extend(
        {"code": "TRANSCRIPTION_WARNING", "message": value} for value in source_warnings
    )

    if not isinstance(answer, dict):
        return {
            "status": "invalid",
            "errors": [{"code": "ANSWER_NOT_OBJECT"}],
            "warnings": warnings,
            "metrics": {"block_count": 0, "unresolved_group_count": 0},
        }
    if set(answer) != {"vat_evidence_blocks", "unresolved_candidate_rows"}:
        errors.append({"code": "TOP_LEVEL_FIELDS_INVALID"})
    blocks = answer.get("vat_evidence_blocks")
    unresolved = answer.get("unresolved_candidate_rows")
    if not isinstance(blocks, list):
        errors.append({"code": "VAT_BLOCKS_NOT_ARRAY"})
        blocks = []
    if not isinstance(unresolved, list):
        errors.append({"code": "UNRESOLVED_ROWS_NOT_ARRAY"})
        unresolved = []
    if len(blocks) > 100:
        errors.append({"code": "TOO_MANY_VAT_BLOCKS"})
    if len(unresolved) > 50:
        errors.append({"code": "TOO_MANY_UNRESOLVED_GROUPS"})

    used_value_rows: set[str] = set()
    for index, block in enumerate(blocks):
        location = f"vat_evidence_blocks[{index}]"
        if not isinstance(block, dict):
            errors.append({"code": "VAT_BLOCK_NOT_OBJECT", "location": location})
            continue
        if set(block) != {"context_rows", "source_row", "row_label", "fields"}:
            errors.append({"code": "VAT_BLOCK_FIELDS_INVALID", "location": location})
        context_rows = block.get("context_rows")
        if not isinstance(context_rows, list) or len(context_rows) > 8:
            errors.append({"code": "INVALID_CONTEXT_ROWS", "location": f"{location}.context_rows"})
            context_rows = []
        if len(context_rows) != len(set(context_rows)):
            errors.append({"code": "DUPLICATE_CONTEXT_ROW", "location": f"{location}.context_rows"})
        for row_id in context_rows:
            if not isinstance(row_id, str) or row_id not in source:
                errors.append(
                    {
                        "code": "UNKNOWN_CONTEXT_ROW",
                        "location": f"{location}.context_rows",
                        "value": row_id,
                    }
                )

        source_row = block.get("source_row")
        if not isinstance(source_row, str) or source_row not in source:
            errors.append(
                {
                    "code": "UNKNOWN_SOURCE_ROW",
                    "location": f"{location}.source_row",
                    "value": source_row,
                }
            )
            row_text = ""
        else:
            row_text = source[source_row]
            if source_row in used_value_rows:
                errors.append(
                    {
                        "code": "SOURCE_ROW_REUSED",
                        "location": f"{location}.source_row",
                        "value": source_row,
                    }
                )
            used_value_rows.add(source_row)

        row_label = block.get("row_label")
        if row_label is not None:
            if not isinstance(row_label, str) or not row_label.strip():
                errors.append({"code": "INVALID_ROW_LABEL", "location": f"{location}.row_label"})
            elif row_label not in row_text:
                errors.append(
                    {
                        "code": "ROW_LABEL_NOT_LITERAL",
                        "location": f"{location}.row_label",
                        "value": row_label,
                    }
                )

        fields = block.get("fields")
        if not isinstance(fields, list) or not 1 <= len(fields) <= 4:
            errors.append({"code": "INVALID_FIELDS", "location": f"{location}.fields"})
            fields = []
        roles: set[str] = set()
        for field_index, field in enumerate(fields):
            field_location = f"{location}.fields[{field_index}]"
            if not isinstance(field, dict) or set(field) != {"role", "value"}:
                errors.append({"code": "VAT_FIELD_INVALID", "location": field_location})
                continue
            role = field.get("role")
            value = field.get("value")
            if role not in _ROLES:
                errors.append(
                    {
                        "code": "VAT_ROLE_INVALID",
                        "location": f"{field_location}.role",
                        "value": role,
                    }
                )
            elif role in roles:
                errors.append(
                    {
                        "code": "VAT_ROLE_REPEATED",
                        "location": f"{field_location}.role",
                        "value": role,
                    }
                )
            else:
                roles.add(role)
            if not isinstance(value, str) or not value.strip():
                errors.append({"code": "VAT_VALUE_INVALID", "location": f"{field_location}.value"})
            elif value not in row_text:
                errors.append(
                    {
                        "code": "VAT_VALUE_NOT_LITERAL",
                        "location": f"{field_location}.value",
                        "value": value,
                    }
                )
            elif parse_decimal_literal(value, percent=(role == "rate_percent")) is None:
                errors.append(
                    {
                        "code": "VAT_VALUE_NOT_PARSEABLE",
                        "location": f"{field_location}.value",
                        "value": value,
                    }
                )

    for index, group in enumerate(unresolved):
        location = f"unresolved_candidate_rows[{index}]"
        if not isinstance(group, dict) or set(group) != {"context_rows", "source_rows"}:
            errors.append({"code": "UNRESOLVED_GROUP_INVALID", "location": location})
            continue
        for key, maximum in (("context_rows", 8), ("source_rows", 8)):
            values = group.get(key)
            if (
                not isinstance(values, list)
                or (key == "source_rows" and not values)
                or len(values) > maximum
            ):
                errors.append({"code": "UNRESOLVED_ROWS_INVALID", "location": f"{location}.{key}"})
                continue
            if len(values) != len(set(values)):
                errors.append(
                    {"code": "UNRESOLVED_ROWS_DUPLICATE", "location": f"{location}.{key}"}
                )
            for row_id in values:
                if not isinstance(row_id, str) or row_id not in source:
                    errors.append(
                        {
                            "code": "UNKNOWN_UNRESOLVED_ROW",
                            "location": f"{location}.{key}",
                            "value": row_id,
                        }
                    )

    return {
        "status": "invalid" if errors else "valid",
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "block_count": len(blocks),
            "unresolved_group_count": len(unresolved),
            "value_row_count": len(used_value_rows),
        },
    }


def _currency(receipt: dict[str, Any]) -> str | None:
    tax = receipt.get("tax")
    tax = tax if isinstance(tax, dict) else {}
    vat_amount = tax.get("vat_amount")
    if isinstance(vat_amount, dict) and isinstance(vat_amount.get("currency"), str):
        return vat_amount["currency"]
    metadata = receipt.get("receipt_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("currency"), str):
        return metadata["currency"]
    return None


def _field_map(block: dict[str, Any]) -> dict[str, Decimal]:
    parsed: dict[str, Decimal] = {}
    for field in block.get("fields") or []:
        if not isinstance(field, dict):
            continue
        role = field.get("role")
        value = parse_decimal_literal(field.get("value"), percent=(role == "rate_percent"))
        if role in _ROLES and value is not None:
            parsed[str(role)] = value
    return parsed


def build_vat_patch(
    source_answer: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    blocks = source_answer.get("vat_evidence_blocks")
    blocks = blocks if isinstance(blocks, list) else []
    vat_lines: list[dict[str, Any]] = []
    aggregate_candidates: list[tuple[Decimal, str]] = []
    ignored: list[dict[str, Any]] = []

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        values = _field_map(block)
        row_label = block.get("row_label")
        source_row = block.get("source_row")
        context_rows = block.get("context_rows") or []
        source_rows = list(dict.fromkeys([*context_rows, source_row]))

        is_aggregate = (
            isinstance(row_label, str)
            and bool(row_label.strip())
            and "vat_amount" in values
            and "rate_percent" not in values
        )
        if is_aggregate:
            aggregate_candidates.append((values["vat_amount"], str(source_row)))
            continue

        if "net_amount" in values and "vat_amount" in values:
            vat_lines.append(
                {
                    "source_rows": source_rows,
                    "rate_percent": (
                        float(values["rate_percent"]) if "rate_percent" in values else None
                    ),
                    "net_amount": money_float(values["net_amount"]),
                    "vat_amount": money_float(values["vat_amount"]),
                }
            )
        else:
            ignored.append(
                {
                    "index": index,
                    "source_row": source_row,
                    "reason": "block_lacks_explicit_net_and_vat_pair",
                    "roles": sorted(values),
                }
            )

    aggregate_value: Decimal | None = None
    if aggregate_candidates:
        distinct = {value for value, _ in aggregate_candidates}
        if len(distinct) > 1:
            return {"patches": []}, {
                "status": "abstained",
                "reason": "conflicting_explicit_aggregate_vat_values",
                "candidates": [
                    {"value": money_float(value), "source_row": row}
                    for value, row in aggregate_candidates
                ],
            }
        aggregate_value = aggregate_candidates[0][0]

    patches: list[dict[str, Any]] = []
    tax = receipt.get("tax")
    tax = tax if isinstance(tax, dict) else {}
    if vat_lines:
        patches.append(
            {
                "op": "replace_value",
                "reason": "Rebuild VAT lines from literal single-row VAT source evidence.",
                "path": "/tax/vat_lines",
                "value": vat_lines,
            }
        )
    if aggregate_value is not None:
        current = tax.get("vat_amount")
        if isinstance(current, dict) and "vat_amount" in current:
            path = "/tax/vat_amount/vat_amount"
            value: Any = money_float(aggregate_value)
        else:
            path = "/tax/vat_amount"
            value = {
                "vat_amount": money_float(aggregate_value),
                "currency": _currency(receipt),
            }
        patches.append(
            {
                "op": "replace_value",
                "reason": "Use the explicitly labelled aggregate VAT amount from source evidence.",
                "path": path,
                "value": value,
            }
        )

    if not patches:
        return {"patches": []}, {
            "status": "abstained",
            "reason": "no_complete_source_supported_vat_mutation",
            "ignored_blocks": ignored,
        }
    return {"patches": patches}, {
        "status": "patch_built",
        "vat_line_count": len(vat_lines),
        "aggregate_vat_replaced": aggregate_value is not None,
        "ignored_blocks": ignored,
    }
