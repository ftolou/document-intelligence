from __future__ import annotations

import json
from dataclasses import dataclass

from receipt_intelligence.application.ports.llm import GenerationRequest, GenerationResult
from receipt_intelligence.extraction.categorization.items import (
    _categorization_output_schema,
    categorize_receipt_items_llm,
)
from receipt_intelligence.extraction.contracts.presentation import (
    CategorizationRequest,
    CategorizationStatus,
)
from receipt_intelligence.extraction.presentation.categorization import (
    ReceiptCategorizationAdapter,
)


@dataclass
class FakeGateway:
    def generate(self, request):  # pragma: no cover - categorizer stub owns the call
        raise AssertionError("gateway should not be called directly")


class CapturingGateway:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(text=json.dumps(self.response))


def test_categorization_without_provider_formatting_still_validates_locally() -> None:
    schema = _categorization_output_schema()
    properties = schema["properties"]
    gateway = CapturingGateway(
        {
            "schema_version": properties["schema_version"]["const"],
            "taxonomy_version": properties["taxonomy_version"]["const"],
            "merchant_taxonomy_version": properties["merchant_taxonomy_version"]["const"],
            "merchant_classification": {
                "category_key": "unknown",
                "confidence": 0.0,
                "reason": "Merchant type is not explicit.",
            },
            "items": [
                {
                    "item_index": 0,
                    "category_key": "unknown",
                    "confidence": 0.0,
                    "text_certainty": "ambiguous",
                    "evidence_terms": ["Milk"],
                    "reason": "The product category is ambiguous.",
                }
            ],
            "warnings": [],
        }
    )

    result = categorize_receipt_items_llm(
        {"items": [{"description": "Milk", "line_total": 1.29}]},
        ollama_url="http://ollama.test",
        model="test-model",
        format_json=False,
        llm_gateway=gateway,
    )

    assert result["status"] in {"ok", "ok_with_warnings"}
    assert gateway.requests[0].format_json is False
    assert gateway.requests[0].response_json_schema is None


def test_disabled_categorization_keeps_receipt_unchanged() -> None:
    receipt = {
        "items": [{"description": "Milk", "line_total": 1.29}],
        "totals": {"grand_total": 1.29},
    }
    service = ReceiptCategorizationAdapter(
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

    service = ReceiptCategorizationAdapter(
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
