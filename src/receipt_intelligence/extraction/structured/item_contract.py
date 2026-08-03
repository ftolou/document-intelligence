"""Read-only contract diagnostics for direct Gemma item output."""

from __future__ import annotations

from typing import Any


def validate_direct_items(answer: Any) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "item_count": 0,
        "items_with_price": 0,
        "items_with_quantity": 0,
        "items_with_unit": 0,
        "items_with_discount_amount": 0,
        "items_with_original_price": 0,
        "extracted_price_sum": None,
    }
    if not isinstance(answer, dict):
        errors.append(
            {"code": "ANSWER_NOT_OBJECT", "message": "The direct item answer is not a JSON object."}
        )
        return _result(errors, warnings, observations, metrics)
    items = answer.get("items")
    if not isinstance(items, list):
        errors.append({"code": "ITEMS_NOT_ARRAY", "message": "The items field is not an array."})
        return _result(errors, warnings, observations, metrics)

    price_sum = 0.0
    seen: set[tuple[str, float | None, float | None, str | None]] = set()
    for index, item in enumerate(items):
        location = f"items[{index}]"
        if not isinstance(item, dict):
            errors.append(
                {
                    "code": "ITEM_NOT_OBJECT",
                    "location": location,
                    "message": "Item is not an object.",
                }
            )
            continue
        name = item.get("name")
        final_price = item.get("final_price")
        quantity = item.get("quantity")
        unit = item.get("unit")
        discount_amount = item.get("discount_amount")
        original_price = item.get("original_price")
        if not isinstance(name, str) or not name.strip():
            errors.append(
                {
                    "code": "INVALID_ITEM_NAME",
                    "location": f"{location}.name",
                    "message": "Item name must be a non-empty string.",
                }
            )
            normalized_name = ""
        else:
            normalized_name = " ".join(name.casefold().split())
        if final_price is not None and (
            isinstance(final_price, bool) or not isinstance(final_price, (int, float))
        ):
            errors.append(
                {
                    "code": "INVALID_FINAL_PRICE_TYPE",
                    "location": f"{location}.final_price",
                    "message": "final_price must be a number or null.",
                }
            )
        elif isinstance(final_price, (int, float)):
            if final_price < 0:
                errors.append(
                    {
                        "code": "NEGATIVE_FINAL_PRICE",
                        "location": f"{location}.final_price",
                        "message": "final_price cannot be negative.",
                    }
                )
            else:
                metrics["items_with_price"] += 1
                price_sum += float(final_price)
        else:
            warnings.append(
                {
                    "code": "MISSING_FINAL_PRICE",
                    "location": f"{location}.final_price",
                    "message": "The model returned no final price for this item.",
                }
            )
        if quantity is not None and (
            isinstance(quantity, bool) or not isinstance(quantity, (int, float))
        ):
            errors.append(
                {
                    "code": "INVALID_QUANTITY_TYPE",
                    "location": f"{location}.quantity",
                    "message": "quantity must be a number or null.",
                }
            )
        elif isinstance(quantity, (int, float)):
            if quantity < 0:
                errors.append(
                    {
                        "code": "NEGATIVE_QUANTITY",
                        "location": f"{location}.quantity",
                        "message": "quantity cannot be negative.",
                    }
                )
            else:
                metrics["items_with_quantity"] += 1
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            errors.append(
                {
                    "code": "INVALID_UNIT",
                    "location": f"{location}.unit",
                    "message": "unit must be a non-empty string or null.",
                }
            )
        elif isinstance(unit, str):
            metrics["items_with_unit"] += 1
        for field_name, value, metric_name in (
            ("discount_amount", discount_amount, "items_with_discount_amount"),
            ("original_price", original_price, "items_with_original_price"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                errors.append(
                    {
                        "code": f"INVALID_{field_name.upper()}_TYPE",
                        "location": f"{location}.{field_name}",
                        "message": f"{field_name} must be a number or null.",
                    }
                )
            elif isinstance(value, (int, float)):
                if value < 0:
                    errors.append(
                        {
                            "code": f"NEGATIVE_{field_name.upper()}",
                            "location": f"{location}.{field_name}",
                            "message": f"{field_name} cannot be negative.",
                        }
                    )
                else:
                    metrics[metric_name] += 1
        if quantity is None and unit is not None:
            warnings.append(
                {
                    "code": "UNIT_WITHOUT_QUANTITY",
                    "location": location,
                    "message": "A unit was returned while quantity is null.",
                }
            )
        key = (
            normalized_name,
            float(final_price)
            if isinstance(final_price, (int, float)) and not isinstance(final_price, bool)
            else None,
            float(quantity)
            if isinstance(quantity, (int, float)) and not isinstance(quantity, bool)
            else None,
            unit.casefold().strip() if isinstance(unit, str) else None,
        )
        if normalized_name and key in seen:
            observations.append(
                {
                    "code": "EXACT_DUPLICATE_ITEM_OBJECT",
                    "location": location,
                    "message": "An identical item object appears more than once; repeated purchases are allowed.",
                }
            )
        seen.add(key)
    metrics["item_count"] = len(items)
    metrics["extracted_price_sum"] = round(price_sum, 2)
    return _result(errors, warnings, observations, metrics)


def _result(
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "invalid" if errors else ("valid_with_warnings" if warnings else "valid"),
        "errors": errors,
        "warnings": warnings,
        "observations": observations,
        "metrics": metrics,
    }


__all__ = ["validate_direct_items"]
