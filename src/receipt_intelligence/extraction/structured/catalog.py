"""Registry-backed Gemma task catalog. Prompt text and schemas remain external."""

from __future__ import annotations

from dataclasses import dataclass

from receipt_intelligence.prompts.registry import PromptReference

SYSTEM_PROMPT = PromptReference("gemma.system.receipt_interpreter", "1.0.0")
TASK_ENVELOPE = PromptReference("gemma.template.task_envelope", "1.0.0")


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    task_name: str
    prompt: PromptReference
    num_predict: int


SCALAR_TASKS = {
    "merchant_name": TaskDefinition(
        "merchant_name", PromptReference("gemma.scalar.merchant_name", "1.0.0"), 96
    ),
    "merchant_address": TaskDefinition(
        "merchant_address", PromptReference("gemma.scalar.merchant_address", "1.0.0"), 192
    ),
    "receipt_date": TaskDefinition(
        "receipt_date", PromptReference("gemma.scalar.receipt_date", "1.0.0"), 64
    ),
    "receipt_time": TaskDefinition(
        "receipt_time", PromptReference("gemma.scalar.receipt_time", "1.0.0"), 64
    ),
    "receipt_number": TaskDefinition(
        "receipt_number", PromptReference("gemma.scalar.receipt_number", "1.0.0"), 96
    ),
    "currency": TaskDefinition("currency", PromptReference("gemma.scalar.currency", "1.0.0"), 48),
    "final_purchase_total": TaskDefinition(
        "final_purchase_total", PromptReference("gemma.scalar.final_purchase_total", "1.0.0"), 128
    ),
    "pre_discount_total": TaskDefinition(
        "pre_discount_total", PromptReference("gemma.scalar.pre_discount_total", "1.0.0"), 128
    ),
    "discount_total": TaskDefinition(
        "discount_total", PromptReference("gemma.scalar.discount_total", "1.0.0"), 128
    ),
    "payment_method": TaskDefinition(
        "payment_method", PromptReference("gemma.scalar.payment_method", "1.0.0"), 64
    ),
    "payment_received": TaskDefinition(
        "payment_received", PromptReference("gemma.scalar.payment_received", "1.0.0"), 128
    ),
    "change_returned": TaskDefinition(
        "change_returned", PromptReference("gemma.scalar.change_returned", "1.0.0"), 128
    ),
    "transaction_status": TaskDefinition(
        "transaction_status", PromptReference("gemma.scalar.transaction_status", "1.0.0"), 48
    ),
    "net_amount": TaskDefinition(
        "net_amount", PromptReference("gemma.scalar.net_amount", "1.0.0"), 128
    ),
    "vat_amount": TaskDefinition(
        "vat_amount", PromptReference("gemma.scalar.vat_amount", "1.0.0"), 128
    ),
    "vat_lines": TaskDefinition(
        "vat_lines", PromptReference("gemma.scalar.vat_lines", "1.0.0"), 768
    ),
}

ITEM_TASK = TaskDefinition(
    "direct_receipt_items",
    PromptReference("gemma.items.direct", "1.0.0"),
    4096,
)


__all__ = ["ITEM_TASK", "SCALAR_TASKS", "SYSTEM_PROMPT", "TASK_ENVELOPE", "TaskDefinition"]
