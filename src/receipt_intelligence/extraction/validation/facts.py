"""Shared immutable facts derived once for independent validation rules."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from receipt_intelligence.extraction.contracts.validation import ValidationRequest

MONEY_QUANTUM = Decimal("0.01")


def decimal_number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value))
    except Exception:
        return None
    return number if number.is_finite() else None


def money(value: Any) -> Decimal | None:
    number = decimal_number(value)
    if number is None:
        return None
    return number.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def money_float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def normalize_currency(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if not normalized or normalized in {"N/A", "NA", "NONE", "NULL", "UNKNOWN", "UNK", "-"}:
        return None
    aliases = {"€": "EUR", "EURO": "EUR", "$": "USD", "US$": "USD", "£": "GBP", "¥": "JPY"}
    return aliases.get(normalized, normalized)


def named_money_value(payload: Any, field_name: str) -> Decimal | None:
    if not isinstance(payload, dict):
        return None
    return money(payload.get(field_name))


@dataclass(frozen=True, slots=True)
class ValidationFacts:
    request: ValidationRequest
    items_value: Any
    items: tuple[Any, ...]
    totals: dict[str, Any]
    discount_section: dict[str, Any]
    payment: dict[str, Any]
    tax: dict[str, Any]
    metadata: dict[str, Any]
    final_total_payload: Any
    pre_discount_payload: Any
    net_amount_payload: Any
    discount_payload: Any
    payment_received_payload: Any
    change_payload: Any
    vat_amount_payload: Any
    final_total: Decimal | None
    pre_discount_total: Decimal | None
    net_amount: Decimal | None
    discount_total: Decimal | None
    payment_received: Decimal | None
    change_returned: Decimal | None
    vat_amount: Decimal | None
    numeric_item_prices: tuple[Decimal, ...]
    missing_price_indices: tuple[int, ...]
    item_discount_failures: tuple[dict[str, Any], ...]
    duplicate_groups: tuple[tuple[int, ...], ...]
    item_sum: Decimal | None
    vat_lines: tuple[Any, ...]
    line_vat_values: tuple[Decimal, ...]
    line_net_values: tuple[Decimal, ...]
    incomplete_vat_line_indices: tuple[int, ...]
    vat_rate_failures: tuple[dict[str, Any], ...]
    currency_sources: tuple[dict[str, str], ...]

    @classmethod
    def build(cls, request: ValidationRequest) -> ValidationFacts:
        receipt = request.receipt
        items_value = receipt.get("items")
        items = tuple(items_value) if isinstance(items_value, list) else ()
        totals = receipt.get("totals") if isinstance(receipt.get("totals"), dict) else {}
        discount_section = (
            receipt.get("discount") if isinstance(receipt.get("discount"), dict) else {}
        )
        payment = receipt.get("payment") if isinstance(receipt.get("payment"), dict) else {}
        tax = receipt.get("tax") if isinstance(receipt.get("tax"), dict) else {}
        metadata = (
            receipt.get("receipt_metadata")
            if isinstance(receipt.get("receipt_metadata"), dict)
            else {}
        )
        final_total_payload = totals.get("final_purchase_total")
        pre_discount_payload = totals.get("pre_discount_total")
        net_amount_payload = totals.get("net_amount")
        discount_payload = discount_section.get("discount_total")
        payment_received_payload = payment.get("payment_received")
        change_payload = payment.get("change_returned")
        vat_amount_payload = tax.get("vat_amount")
        final_total = named_money_value(final_total_payload, "final_purchase_total")
        pre_discount_total = named_money_value(pre_discount_payload, "pre_discount_total")
        net_amount = named_money_value(net_amount_payload, "net_amount")
        discount_total = named_money_value(discount_payload, "discount_total")
        payment_received = named_money_value(payment_received_payload, "payment_received")
        change_returned = named_money_value(change_payload, "change_returned")
        vat_amount = named_money_value(vat_amount_payload, "vat_amount")

        numeric_item_prices: list[Decimal] = []
        missing_price_indices: list[int] = []
        item_discount_failures: list[dict[str, Any]] = []
        duplicate_keys: dict[tuple[str, Decimal | None, Decimal | None, str | None], list[int]] = {}
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            final_price = money(item.get("final_price"))
            original_price = money(item.get("original_price"))
            item_discount = money(item.get("discount_amount"))
            quantity = decimal_number(item.get("quantity"))
            unit = item.get("unit")
            name = item.get("name")
            if final_price is None:
                missing_price_indices.append(index)
            else:
                numeric_item_prices.append(final_price)
            if original_price is not None and final_price is not None and item_discount is not None:
                expected = original_price - item_discount
                if not request_money_close(request, expected, final_price):
                    item_discount_failures.append(
                        {
                            "item_index": index,
                            "name": name,
                            "original_price": money_float(original_price),
                            "discount_amount": money_float(item_discount),
                            "expected_final_price": money_float(expected),
                            "final_price": money_float(final_price),
                            "difference": money_float(final_price - expected),
                            "reason": "original_minus_discount_does_not_equal_final",
                        }
                    )
            elif (
                original_price is not None
                and final_price is not None
                and item_discount is None
                and not request_money_close(request, original_price, final_price)
            ):
                item_discount_failures.append(
                    {
                        "item_index": index,
                        "name": name,
                        "original_price": money_float(original_price),
                        "discount_amount": None,
                        "final_price": money_float(final_price),
                        "difference": money_float(original_price - final_price),
                        "reason": "price_changed_without_discount_amount",
                    }
                )
            elif item_discount is not None and original_price is None:
                item_discount_failures.append(
                    {
                        "item_index": index,
                        "name": name,
                        "original_price": None,
                        "discount_amount": money_float(item_discount),
                        "final_price": money_float(final_price),
                        "reason": "discount_amount_without_original_price",
                    }
                )
            normalized_name = (
                " ".join(name.casefold().split()) if isinstance(name, str) and name.strip() else ""
            )
            duplicate_key = (
                normalized_name,
                final_price,
                quantity,
                unit.casefold().strip() if isinstance(unit, str) else None,
            )
            if normalized_name:
                duplicate_keys.setdefault(duplicate_key, []).append(index)
        duplicate_groups = tuple(
            tuple(indices) for indices in duplicate_keys.values() if len(indices) > 1
        )
        item_sum = (
            sum(numeric_item_prices, Decimal("0.00")).quantize(MONEY_QUANTUM)
            if numeric_item_prices
            else None
        )

        vat_lines_value = tax.get("vat_lines")
        vat_lines = tuple(vat_lines_value) if isinstance(vat_lines_value, list) else ()
        line_vat_values: list[Decimal] = []
        line_net_values: list[Decimal] = []
        incomplete_vat_line_indices: list[int] = []
        vat_rate_failures: list[dict[str, Any]] = []
        for index, line in enumerate(vat_lines):
            if not isinstance(line, dict):
                incomplete_vat_line_indices.append(index)
                continue
            line_rate = decimal_number(line.get("rate_percent"))
            line_net = money(line.get("net_amount"))
            line_vat = money(line.get("vat_amount"))
            if line_vat is not None:
                line_vat_values.append(line_vat)
            if line_net is not None:
                line_net_values.append(line_net)
            if line_net is None or line_vat is None:
                incomplete_vat_line_indices.append(index)
            if line_rate is not None and line_net is not None and line_vat is not None:
                expected_vat = (line_net * line_rate / Decimal("100")).quantize(
                    MONEY_QUANTUM, rounding=ROUND_HALF_UP
                )
                if not request_vat_close(request, expected_vat, line_vat):
                    vat_rate_failures.append(
                        {
                            "vat_line_index": index,
                            "rate_percent": float(line_rate),
                            "net_amount": money_float(line_net),
                            "expected_vat_amount": money_float(expected_vat),
                            "vat_amount": money_float(line_vat),
                            "difference": money_float(line_vat - expected_vat),
                        }
                    )

        currency_sources: list[dict[str, str]] = []
        metadata_currency = normalize_currency(metadata.get("currency"))
        if metadata_currency is not None:
            currency_sources.append(
                {
                    "path": "receipt_metadata.currency",
                    "currency": metadata_currency,
                }
            )
        for path, payload in (
            ("totals.final_purchase_total", final_total_payload),
            ("totals.pre_discount_total", pre_discount_payload),
            ("totals.net_amount", net_amount_payload),
            ("discount.discount_total", discount_payload),
            ("payment.payment_received", payment_received_payload),
            ("payment.change_returned", change_payload),
            ("tax.vat_amount", vat_amount_payload),
        ):
            if not isinstance(payload, dict):
                continue
            currency = normalize_currency(payload.get("currency"))
            if currency is not None:
                currency_sources.append({"path": path, "currency": currency})

        return cls(
            request=request,
            items_value=items_value,
            items=items,
            totals=totals,
            discount_section=discount_section,
            payment=payment,
            tax=tax,
            metadata=metadata,
            final_total_payload=final_total_payload,
            pre_discount_payload=pre_discount_payload,
            net_amount_payload=net_amount_payload,
            discount_payload=discount_payload,
            payment_received_payload=payment_received_payload,
            change_payload=change_payload,
            vat_amount_payload=vat_amount_payload,
            final_total=final_total,
            pre_discount_total=pre_discount_total,
            net_amount=net_amount,
            discount_total=discount_total,
            payment_received=payment_received,
            change_returned=change_returned,
            vat_amount=vat_amount,
            numeric_item_prices=tuple(numeric_item_prices),
            missing_price_indices=tuple(missing_price_indices),
            item_discount_failures=tuple(item_discount_failures),
            duplicate_groups=duplicate_groups,
            item_sum=item_sum,
            vat_lines=vat_lines,
            line_vat_values=tuple(line_vat_values),
            line_net_values=tuple(line_net_values),
            incomplete_vat_line_indices=tuple(incomplete_vat_line_indices),
            vat_rate_failures=tuple(vat_rate_failures),
            currency_sources=tuple(currency_sources),
        )


def request_money_close(request: ValidationRequest, first: Decimal, second: Decimal) -> bool:
    return abs(first - second) <= Decimal(str(request.money_tolerance))


def request_vat_close(request: ValidationRequest, first: Decimal, second: Decimal) -> bool:
    return abs(first - second) <= Decimal(str(request.vat_rate_tolerance))


__all__ = [
    "MONEY_QUANTUM",
    "ValidationFacts",
    "money_float",
    "request_money_close",
]
