from __future__ import annotations

from receipt_intelligence.prompts.registry import PromptReference, default_prompt_registry


def test_all_phase3_gemma_prompts_and_schemas_verify() -> None:
    registry = default_prompt_registry()
    required = [
        "gemma.system.receipt_interpreter",
        "gemma.template.task_envelope",
        "gemma.items.direct",
        "gemma.scalar.merchant_name",
        "gemma.scalar.merchant_address",
        "gemma.scalar.currency",
        "gemma.scalar.final_purchase_total",
        "gemma.scalar.discount_total",
        "gemma.scalar.vat_amount",
        "gemma.scalar.vat_lines",
    ]
    for prompt_id in required:
        reference = PromptReference(prompt_id, "1.0.0")
        assert registry.read_template(reference).strip()
        if prompt_id not in {"gemma.system.receipt_interpreter", "gemma.template.task_envelope"}:
            assert registry.load_schema(reference) is not None
