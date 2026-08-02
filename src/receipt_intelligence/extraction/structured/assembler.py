"""Pure receipt assembly from completed scalar and item task outputs."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.extraction.contracts.extraction import GemmaTaskResult, GemmaTaskStatus


def assemble_receipt(
    scalar_results: tuple[GemmaTaskResult, ...],
    item_result: GemmaTaskResult | None,
) -> dict[str, Any]:
    by_name = {result.task_name: result for result in scalar_results}

    def answer(task_name: str) -> dict[str, Any] | None:
        result = by_name.get(task_name)
        if result is None or result.status is not GemmaTaskStatus.COMPLETED:
            return None
        return result.answer

    def field(task_name: str, field_name: str) -> Any:
        value = answer(task_name)
        return value.get(field_name) if value else None

    vat_lines = answer("vat_lines")
    items = (
        item_result.answer.get("items")
        if item_result is not None
        and item_result.status is GemmaTaskStatus.COMPLETED
        and item_result.answer is not None
        else None
    )
    return {
        "merchant": {
            "name": field("merchant_name", "merchant_name"),
            "address": answer("merchant_address"),
        },
        "receipt_metadata": {
            "date": field("receipt_date", "receipt_date"),
            "time": field("receipt_time", "receipt_time"),
            "receipt_number": field("receipt_number", "receipt_number"),
            "currency": field("currency", "currency"),
        },
        "items": items,
        "totals": {
            "final_purchase_total": answer("final_purchase_total"),
            "pre_discount_total": answer("pre_discount_total"),
            "net_amount": answer("net_amount"),
        },
        "discount": {"discount_total": answer("discount_total")},
        "payment": {
            "payment_method": field("payment_method", "payment_method"),
            "payment_received": answer("payment_received"),
            "change_returned": answer("change_returned"),
        },
        "transaction_status": field("transaction_status", "transaction_status"),
        "tax": {
            "vat_amount": answer("vat_amount"),
            "vat_lines": vat_lines.get("vat_lines") if vat_lines else None,
        },
    }


__all__ = ["assemble_receipt"]
