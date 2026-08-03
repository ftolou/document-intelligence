from __future__ import annotations

from decimal import Decimal
from typing import Any

from ..evidence import money_float, parse_decimal_literal, row_map


def validate_final_total_evidence(answer: Any, transcription: str) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        source, source_warnings = row_map(transcription)
    except Exception as exc:
        return {
            "status": "invalid",
            "errors": [{"code": "SOURCE_ROWS_UNAVAILABLE", "message": str(exc)}],
            "warnings": [],
        }
    warnings.extend(
        {"code": "TRANSCRIPTION_WARNING", "message": value} for value in source_warnings
    )
    expected_fields = {"status", "label_row", "source_row", "label_text", "value_text"}
    if not isinstance(answer, dict):
        return {
            "status": "invalid",
            "errors": [{"code": "ANSWER_NOT_OBJECT"}],
            "warnings": warnings,
        }
    if set(answer) != expected_fields:
        errors.append({"code": "TOP_LEVEL_FIELDS_INVALID"})
    status = answer.get("status")
    if status not in {"resolved", "unresolved"}:
        errors.append({"code": "STATUS_INVALID", "value": status})
    if status == "unresolved":
        for key in ("label_row", "source_row", "label_text", "value_text"):
            if answer.get(key) is not None:
                errors.append({"code": "UNRESOLVED_FIELD_NOT_NULL", "location": key})
    elif status == "resolved":
        label_row = answer.get("label_row")
        source_row = answer.get("source_row")
        label_text = answer.get("label_text")
        value_text = answer.get("value_text")
        if not isinstance(label_row, str) or label_row not in source:
            errors.append({"code": "LABEL_ROW_INVALID", "value": label_row})
        if not isinstance(source_row, str) or source_row not in source:
            errors.append({"code": "SOURCE_ROW_INVALID", "value": source_row})
        if not isinstance(label_text, str) or not label_text.strip():
            errors.append({"code": "LABEL_TEXT_INVALID"})
        elif (
            isinstance(label_row, str)
            and label_row in source
            and label_text not in source[label_row]
        ):
            errors.append({"code": "LABEL_TEXT_NOT_LITERAL", "value": label_text})
        if not isinstance(value_text, str) or not value_text.strip():
            errors.append({"code": "VALUE_TEXT_INVALID"})
        elif (
            isinstance(source_row, str)
            and source_row in source
            and value_text not in source[source_row]
        ):
            errors.append({"code": "VALUE_TEXT_NOT_LITERAL", "value": value_text})
        parsed = parse_decimal_literal(value_text)
        if parsed is None or parsed < 0:
            errors.append({"code": "VALUE_TEXT_NOT_PARSEABLE", "value": value_text})
    return {"status": "invalid" if errors else "valid", "errors": errors, "warnings": warnings}


def _current_total(receipt: dict[str, Any]) -> Decimal | None:
    totals = receipt.get("totals")
    totals = totals if isinstance(totals, dict) else {}
    payload = totals.get("final_purchase_total")
    if not isinstance(payload, dict):
        return None
    value = payload.get("final_purchase_total")
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except Exception:
        return None


def _currency(receipt: dict[str, Any]) -> str | None:
    totals = receipt.get("totals")
    totals = totals if isinstance(totals, dict) else {}
    payload = totals.get("final_purchase_total")
    if isinstance(payload, dict) and isinstance(payload.get("currency"), str):
        return payload["currency"]
    metadata = receipt.get("receipt_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("currency"), str):
        return metadata["currency"]
    return None


def build_final_total_patch(
    source_answer: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if source_answer.get("status") != "resolved":
        return {"patches": []}, {"status": "abstained", "reason": "source_total_unresolved"}
    parsed = parse_decimal_literal(source_answer.get("value_text"))
    if parsed is None or parsed < 0:
        return {"patches": []}, {"status": "abstained", "reason": "source_total_not_parseable"}
    current = _current_total(receipt)
    if current is not None and current == parsed:
        return {"patches": []}, {
            "status": "abstained",
            "reason": "source_total_matches_current_total",
        }

    totals = receipt.get("totals")
    totals = totals if isinstance(totals, dict) else {}
    current_payload = totals.get("final_purchase_total")
    if isinstance(current_payload, dict) and "final_purchase_total" in current_payload:
        path = "/totals/final_purchase_total/final_purchase_total"
        value: Any = money_float(parsed)
    else:
        path = "/totals/final_purchase_total"
        value = {
            "final_purchase_total": money_float(parsed),
            "currency": _currency(receipt),
        }
    return {
        "patches": [
            {
                "op": "replace_value",
                "reason": (
                    f"Explicit final-purchase-total label {source_answer.get('label_text')!r} "
                    f"at {source_answer.get('label_row')} points to printed amount "
                    f"{source_answer.get('value_text')!r} at {source_answer.get('source_row')}."
                )[:240],
                "path": path,
                "value": value,
            }
        ]
    }, {
        "status": "patch_built",
        "label_row": source_answer.get("label_row"),
        "source_row": source_answer.get("source_row"),
        "label_text": source_answer.get("label_text"),
        "value_text": source_answer.get("value_text"),
    }
