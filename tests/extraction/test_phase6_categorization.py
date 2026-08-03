from __future__ import annotations

from dataclasses import dataclass

from receipt_intelligence.extraction.contracts.presentation import (
    CategorizationRequest,
    CategorizationStatus,
)
from receipt_intelligence.extraction.presentation.categorization import (
    ExistingReceiptCategorizationService,
)


@dataclass
class FakeGateway:
    def generate(self, request):  # pragma: no cover - categorizer stub owns the call
        raise AssertionError("gateway should not be called directly")


def test_disabled_categorization_keeps_receipt_unchanged() -> None:
    receipt = {
        "items": [{"description": "Milk", "line_total": 1.29}],
        "totals": {"grand_total": 1.29},
    }
    service = ExistingReceiptCategorizationService(
        llm_gateway=FakeGateway(),
        ollama_url="http://ollama",
        model="gemma4",
        categorizer=lambda *args, **kwargs: {},
    )
    result = service.categorize(CategorizationRequest(run_id="r1", receipt=receipt, enabled=False))
    assert result.status is CategorizationStatus.DISABLED
    assert result.receipt == receipt
    assert result.receipt is not receipt


def test_categorization_adapter_preserves_math_fields() -> None:
    receipt = {
        "items": [{"description": "Milk", "line_total": 1.29}],
        "totals": {"grand_total": 1.29},
        "taxes": [{"rate": 7, "tax": 0.08}],
        "payments": [{"method": "card", "amount": 1.29}],
    }

    def categorizer(value, **kwargs):
        categorized = {
            **value,
            "items": [
                {
                    **value["items"][0],
                    "line_total": 999.0,
                    "category_key": "groceries_dairy_eggs",
                }
            ],
            "totals": {"grand_total": 999.0},
            "taxes": [],
            "payments": [],
            "categorization": {"item_count": 1, "categorized_count": 1},
        }
        return {
            "status": "ok",
            "receipt": categorized,
            "categories": [{"item_index": 0, "category_key": "groceries_dairy_eggs"}],
            "merchant_classification": {"category_key": "grocery_store"},
            "warnings": [],
            "prompt": "prompt",
            "raw_output": "{}",
            "duration_seconds": 0.1,
        }

    service = ExistingReceiptCategorizationService(
        llm_gateway=FakeGateway(),
        ollama_url="http://ollama",
        model="gemma4",
        categorizer=categorizer,
    )
    result = service.categorize(CategorizationRequest(run_id="r1", receipt=receipt))
    assert result.status is CategorizationStatus.OK
    assert result.receipt["totals"] == receipt["totals"]
    assert result.receipt["taxes"] == receipt["taxes"]
    assert result.receipt["payments"] == receipt["payments"]
    assert result.receipt["items"][0]["line_total"] == 1.29
