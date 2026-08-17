from __future__ import annotations

import json
from pathlib import Path

from receipt_intelligence.application.ports.chat import ChatGenerationResult
from receipt_intelligence.extraction.contracts.extraction import StructuredExtractionRequest
from receipt_intelligence.extraction.contracts.transcription import TranscriptionResult
from receipt_intelligence.extraction.settings import ParsingSettings
from receipt_intelligence.extraction.structured.item_contract import validate_direct_items
from receipt_intelligence.extraction.structured.service import GemmaStructuredExtractionService
from receipt_intelligence.extraction.structured.task_runner import GemmaTaskRunner
from receipt_intelligence.prompts.registry import default_prompt_registry


class FakeChatGateway:
    def generate(self, request):
        operation = request.operation.removeprefix("receipt_")
        answers = {
            "merchant_name": {"merchant_name": "Testmarkt"},
            "currency": {"currency": "EUR"},
            "final_purchase_total": {"final_purchase_total": 3.5, "currency": "EUR"},
            "direct_receipt_items": {
                "items": [
                    {
                        "name": "Milch",
                        "final_price": 3.5,
                        "quantity": None,
                        "unit": None,
                        "discount_amount": None,
                        "original_price": None,
                    }
                ]
            },
        }
        return ChatGenerationResult(text=json.dumps(answers[operation]))


def test_service_assembles_scalars_and_items(tmp_path: Path) -> None:
    settings = ParsingSettings(
        ollama_url="http://ollama",
        model="gemma4",
        scalar_tasks=("merchant_name", "currency", "final_purchase_total"),
    )
    service = GemmaStructuredExtractionService(
        task_runner=GemmaTaskRunner(
            gateway=FakeChatGateway(),
            prompts=default_prompt_registry(),
            settings=settings,
        ),
        settings=settings,
        result_dir=tmp_path,
    )
    result = service.extract(
        StructuredExtractionRequest(
            run_id="receipt-1",
            transcription=TranscriptionResult(
                canonical_text="R0001 :: Testmarkt\nR0002 :: Milch 3,50",
                rows=(),
                crops=(),
                fragments=(),
            ),
        )
    )
    assert result.receipt["merchant"]["name"] == "Testmarkt"
    assert result.receipt["items"][0]["final_price"] == 3.5
    assert result.receipt["totals"]["final_purchase_total"]["currency"] == "EUR"
    assert result.item_contract["status"] == "valid"


def test_item_contract_is_read_only_and_reports_missing_price() -> None:
    answer = {
        "items": [
            {
                "name": "Menu component",
                "final_price": None,
                "quantity": None,
                "unit": None,
                "discount_amount": None,
                "original_price": None,
            }
        ]
    }
    before = json.loads(json.dumps(answer))
    report = validate_direct_items(answer)
    assert answer == before
    assert report["status"] == "valid_with_warnings"
    assert report["warnings"][0]["code"] == "MISSING_FINAL_PRICE"
