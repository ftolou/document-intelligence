"""Compatibility projection between legacy review/storage fields and next receipts.

The extraction result remains canonical.  Review and relational-storage callers use
these helpers instead of each inventing a different set of fallback field names.
"""

from __future__ import annotations

import copy
from typing import Any

JsonObject = dict[str, Any]


def _object(value: Any) -> JsonObject:
    return value if isinstance(value, dict) else {}


def _items(receipt: JsonObject) -> list[JsonObject]:
    value = receipt.get("items")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def scalar_value(value: Any, *field_names: str) -> Any:
    """Read either a scalar or a schema-wrapped scalar result."""

    if not isinstance(value, dict):
        return value
    for name in field_names:
        candidate = value.get(name)
        if candidate is not None and candidate != "":
            return candidate
    for name in ("value", "amount", "total"):
        candidate = value.get(name)
        if candidate is not None and candidate != "":
            return candidate
    return None


def is_next_receipt(receipt: JsonObject) -> bool:
    totals = _object(receipt.get("totals"))
    return bool(
        isinstance(receipt.get("receipt_metadata"), dict)
        or isinstance(totals.get("final_purchase_total"), dict)
        or any("final_price" in item for item in _items(receipt))
    )


def receipt_date(receipt: JsonObject) -> Any:
    return first_present(_object(receipt.get("receipt_metadata")).get("date"), receipt.get("date"))


def receipt_time(receipt: JsonObject) -> Any:
    return first_present(_object(receipt.get("receipt_metadata")).get("time"), receipt.get("time"))


def receipt_number(receipt: JsonObject) -> Any:
    return first_present(
        _object(receipt.get("receipt_metadata")).get("receipt_number"),
        receipt.get("receipt_number"),
    )


def receipt_currency(receipt: JsonObject) -> Any:
    return first_present(
        _object(receipt.get("receipt_metadata")).get("currency"),
        receipt.get("currency"),
    )


def merchant_address_text(receipt: JsonObject) -> str:
    address = _object(receipt.get("merchant")).get("address")
    if address in (None, ""):
        return ""
    if not isinstance(address, dict):
        return str(address).strip()
    formatted = first_present(
        address.get("formatted"),
        address.get("formatted_address"),
        address.get("full_address"),
    )
    if formatted:
        return str(formatted).strip()
    street = " ".join(
        str(part).strip()
        for part in (address.get("street"), address.get("house_number"))
        if part not in (None, "")
    ).strip()
    city = " ".join(
        str(part).strip()
        for part in (address.get("postal_code"), address.get("city"))
        if part not in (None, "")
    ).strip()
    return ", ".join(
        str(part).strip()
        for part in (street, city, address.get("country"))
        if part not in (None, "") and str(part).strip()
    )


def receipt_subtotal(receipt: JsonObject) -> Any:
    totals = _object(receipt.get("totals"))
    return first_present(
        scalar_value(totals.get("subtotal"), "subtotal"),
        scalar_value(totals.get("net_amount"), "net_amount"),
    )


def receipt_tax_total(receipt: JsonObject) -> Any:
    totals = _object(receipt.get("totals"))
    tax = _object(receipt.get("tax"))
    return first_present(
        scalar_value(totals.get("tax_total"), "tax_total"),
        scalar_value(tax.get("vat_amount"), "vat_amount"),
    )


def receipt_grand_total(receipt: JsonObject) -> Any:
    totals = _object(receipt.get("totals"))
    return first_present(
        scalar_value(totals.get("grand_total"), "grand_total"),
        scalar_value(totals.get("final_purchase_total"), "final_purchase_total"),
    )


def receipt_paid_total(receipt: JsonObject) -> Any:
    totals = _object(receipt.get("totals"))
    payment = _object(receipt.get("payment"))
    payments = receipt.get("payments") if isinstance(receipt.get("payments"), list) else []
    legacy_payment = payments[0] if payments and isinstance(payments[0], dict) else {}
    return first_present(
        scalar_value(totals.get("paid_total"), "paid_total"),
        scalar_value(payment.get("payment_received"), "payment_received"),
        legacy_payment.get("amount"),
        receipt_grand_total(receipt),
    )


def receipt_change(receipt: JsonObject) -> Any:
    totals = _object(receipt.get("totals"))
    payment = _object(receipt.get("payment"))
    return first_present(
        scalar_value(totals.get("change"), "change"),
        scalar_value(payment.get("change_returned"), "change_returned"),
    )


def receipt_payment_method(receipt: JsonObject) -> Any:
    payment = _object(receipt.get("payment"))
    payments = receipt.get("payments") if isinstance(receipt.get("payments"), list) else []
    legacy_payment = payments[0] if payments and isinstance(payments[0], dict) else {}
    return first_present(payment.get("payment_method"), legacy_payment.get("method"))


def item_description(item: JsonObject) -> str:
    return str(
        first_present(
            item.get("product_description"),
            item.get("clean_description"),
            item.get("normalized_name"),
            item.get("description"),
            item.get("name"),
            item.get("raw_name"),
            item.get("text"),
            "",
        )
    )


def item_line_total(item: JsonObject) -> Any:
    return first_present(
        item.get("line_total"),
        item.get("final_price"),
        item.get("total"),
        item.get("amount"),
    )


def validation_issues(validation: JsonObject) -> list[JsonObject]:
    issues = validation.get("issues")
    if isinstance(issues, list):
        existing = [copy.deepcopy(item) for item in issues if isinstance(item, dict)]
        if existing:
            return existing
    checks = validation.get("checks")
    if not isinstance(checks, list):
        return []
    return [
        {
            "code": str(check.get("code") or "VALIDATION_FAILED"),
            "severity": str(check.get("severity") or "review"),
            "message": str(check.get("message") or "Validation check failed."),
            "details": copy.deepcopy(check.get("details")),
        }
        for check in checks
        if isinstance(check, dict) and str(check.get("status") or "").lower() == "failed"
    ]


def validation_for_review(validation: JsonObject) -> JsonObject:
    result = copy.deepcopy(validation)
    result["issues"] = validation_issues(validation)
    status = str(result.get("status") or "").strip().lower()
    result.setdefault(
        "import_decision",
        {"valid": "import", "review_required": "needs_review", "invalid": "reject"}.get(
            status, "needs_review"
        ),
    )
    if "balanced" not in result:
        result["balanced"] = not result["issues"]
    metrics = _object(result.get("metrics"))
    if "difference" not in result:
        item_sum = scalar_value(metrics.get("item_sum"), "item_sum")
        total = scalar_value(metrics.get("final_purchase_total"), "final_purchase_total")
        try:
            result["difference"] = round(float(item_sum) - float(total), 2)
        except (TypeError, ValueError):
            result["difference"] = None
    return result


def to_review_document(receipt: JsonObject) -> JsonObject:
    """Return a display/edit projection without mutating the canonical receipt."""

    result = copy.deepcopy(receipt)
    metadata = _object(result.get("receipt_metadata"))
    result["date"] = receipt_date(receipt)
    result["time"] = receipt_time(receipt)
    result["currency"] = receipt_currency(receipt) or "EUR"
    result["receipt_number"] = receipt_number(receipt)
    if metadata:
        result["receipt_metadata"] = metadata

    merchant = dict(_object(result.get("merchant")))
    merchant["address"] = merchant_address_text(receipt)
    result["merchant"] = merchant

    totals = dict(_object(result.get("totals")))
    totals.update(
        {
            "subtotal": receipt_subtotal(receipt),
            "tax_total": receipt_tax_total(receipt),
            "grand_total": receipt_grand_total(receipt),
            "paid_total": receipt_paid_total(receipt),
            "change": receipt_change(receipt),
        }
    )
    result["totals"] = totals
    result["payment_method"] = receipt_payment_method(receipt)

    projected_items: list[JsonObject] = []
    for original in _items(result):
        item = dict(original)
        description = item_description(item)
        if not item.get("description"):
            item["description"] = description
        if not item.get("product_description"):
            item["product_description"] = description
        item["line_total"] = item_line_total(item)
        projected_items.append(item)
    result["items"] = projected_items

    validation = _object(result.get("validation"))
    if validation:
        result["validation"] = validation_for_review(validation)
    return result


def to_legacy_validation_document(receipt: JsonObject) -> JsonObject:
    """Project a next receipt into the legacy read-only validator contract."""

    result = to_review_document(receipt)
    payment_method = receipt_payment_method(receipt)
    paid_total = receipt_paid_total(receipt)
    result["payments"] = (
        [{"method": payment_method, "amount": paid_total}]
        if payment_method is not None or paid_total is not None
        else []
    )
    vat_lines = _object(receipt.get("tax")).get("vat_lines")
    taxes: list[JsonObject] = []
    if isinstance(vat_lines, list):
        for row in vat_lines:
            if not isinstance(row, dict):
                continue
            net = scalar_value(row.get("net_amount"), "net_amount")
            tax = scalar_value(row.get("vat_amount"), "vat_amount")
            try:
                gross = round(float(net) + float(tax), 2)
            except (TypeError, ValueError):
                gross = None
            taxes.append(
                {
                    "rate": row.get("rate_percent"),
                    "net": net,
                    "tax": tax,
                    "gross": gross,
                    "source_line_ids": copy.deepcopy(row.get("source_rows") or []),
                }
            )
    result["taxes"] = taxes
    return result


def _set_wrapped(container: JsonObject, key: str, field: str, value: Any, currency: Any) -> None:
    existing = container.get(key)
    if isinstance(existing, dict):
        wrapped = dict(existing)
        wrapped[field] = value
        if "currency" in wrapped or currency is not None:
            wrapped["currency"] = currency
        container[key] = wrapped
    else:
        container[key] = {field: value, "currency": currency}


def apply_review_field(receipt: JsonObject, key: str, value: Any) -> bool:
    """Apply one UI header field to the canonical schema in-place."""

    next_schema = is_next_receipt(receipt)
    merchant = receipt.setdefault("merchant", {})
    if not isinstance(merchant, dict):
        merchant = {}
        receipt["merchant"] = merchant
    if key == "merchant_name":
        merchant["name"] = value
        return True
    if key == "merchant_address":
        if next_schema:
            merchant["address"] = {"formatted": value} if value not in (None, "") else None
        else:
            merchant["address"] = value
        return True
    if key in {"document_type", "receipt_category", "receipt_business_category"}:
        receipt[key] = value
        return True
    if not next_schema:
        if key in {"date", "time", "currency", "receipt_number"}:
            receipt[key] = value
            return True
        if key in {"subtotal", "tax_total", "grand_total", "paid_total", "change"}:
            totals = receipt.setdefault("totals", {})
            if isinstance(totals, dict):
                totals[key] = value
                return True
        if key == "payment_method":
            payments = receipt.setdefault("payments", [])
            if not isinstance(payments, list):
                payments = []
                receipt["payments"] = payments
            if not payments or not isinstance(payments[0], dict):
                payments.insert(0, {})
            payments[0]["method"] = value
            return True
        return False

    metadata = receipt.setdefault("receipt_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        receipt["receipt_metadata"] = metadata
    if key in {"date", "time", "currency", "receipt_number"}:
        metadata[key] = value
        if key == "currency":
            for envelope in (
                _object(receipt.get("totals")).values(),
                _object(receipt.get("discount")).values(),
                _object(receipt.get("payment")).values(),
                _object(receipt.get("tax")).values(),
            ):
                for candidate in envelope:
                    if isinstance(candidate, dict) and "currency" in candidate:
                        candidate["currency"] = value
        return True

    currency = receipt_currency(receipt)
    if key == "subtotal":
        totals = receipt.setdefault("totals", {})
        if isinstance(totals, dict):
            _set_wrapped(totals, "net_amount", "net_amount", value, currency)
            return True
    if key == "tax_total":
        tax = receipt.setdefault("tax", {})
        if isinstance(tax, dict):
            _set_wrapped(tax, "vat_amount", "vat_amount", value, currency)
            return True
    if key == "grand_total":
        totals = receipt.setdefault("totals", {})
        if isinstance(totals, dict):
            _set_wrapped(
                totals,
                "final_purchase_total",
                "final_purchase_total",
                value,
                currency,
            )
            return True
    if key == "paid_total":
        payment = receipt.setdefault("payment", {})
        if isinstance(payment, dict):
            _set_wrapped(payment, "payment_received", "payment_received", value, currency)
            return True
    if key == "change":
        payment = receipt.setdefault("payment", {})
        if isinstance(payment, dict):
            _set_wrapped(payment, "change_returned", "change_returned", value, currency)
            return True
    if key == "payment_method":
        payment = receipt.setdefault("payment", {})
        if isinstance(payment, dict):
            payment["payment_method"] = value
            return True
    return False


def apply_review_item_field(item: JsonObject, key: str, value: Any, *, next_schema: bool) -> str:
    """Apply one item form field and return the canonical changed-field name."""

    if next_schema:
        if key in {"description", "product_description"}:
            item["name"] = value
            return "name"
        if key == "line_total":
            item["final_price"] = value
            return "final_price"
        item[key] = value
        return key

    if key == "description":
        item["description"] = value
        item.setdefault("raw_name", value)
        return "description"
    if key == "product_description":
        item["product_description"] = value
        if not item.get("description"):
            item["description"] = value
        return "product_description"
    if key == "parser_item_type":
        item["category"] = value
        item["parser_item_type"] = value
        return "parser_item_type"
    item[key] = value
    return key


__all__ = [
    "apply_review_field",
    "apply_review_item_field",
    "first_present",
    "is_next_receipt",
    "item_description",
    "item_line_total",
    "merchant_address_text",
    "receipt_change",
    "receipt_currency",
    "receipt_date",
    "receipt_grand_total",
    "receipt_number",
    "receipt_paid_total",
    "receipt_payment_method",
    "receipt_subtotal",
    "receipt_tax_total",
    "receipt_time",
    "scalar_value",
    "to_legacy_validation_document",
    "to_review_document",
    "validation_for_review",
    "validation_issues",
]
