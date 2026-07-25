from __future__ import annotations

import copy
import json
from dataclasses import fields
from pathlib import Path

from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports import OcrRequest, VlmRequest
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    GenerationResult,
    ModelCallMetrics,
)
from receipt_intelligence.extraction.parsing.llm_parser import run_llm_main_parser


class FakeLlmGateway:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(text=self.responses.pop(0))


def _valid_receipt() -> dict[str, object]:
    return {
        "schema_version": "v14_6_llm_receipt_1",
        "parse_status": "ok",
        "currency": "EUR",
        "merchant": {
            "name": "TEST",
            "address": None,
            "tax_id": None,
            "source_line_ids": [],
        },
        "date": None,
        "time": None,
        "items": [],
        "taxes": [],
        "totals": {
            "subtotal": None,
            "tax_total": None,
            "grand_total": None,
            "paid_total": None,
            "change": None,
            "source_line_ids": [],
        },
        "payments": [],
        "unresolved_rows": [],
        "warnings": [],
        "overall_confidence": 0.8,
    }


def test_generation_result_deepcopy_is_a_normal_value_object() -> None:
    metrics = ModelCallMetrics(
        provider="ollama",
        endpoint="generate",
        model="gemma4",
        request_duration_ms=12.5,
    )
    original = GenerationResult(text='{"ok":true}', metrics=metrics)

    copied = copy.deepcopy(original)

    assert copied == original
    assert copied.text == '{"ok":true}'
    assert copied.metrics == metrics


def test_json_parser_accepts_explicit_generation_result() -> None:
    result = GenerationResult(text='```json\n{"status":"ok"}\n```')

    assert parse_json_from_llm(result) == {"status": "ok"}


def test_extraction_main_parser_accepts_fake_gateway(tmp_path: Path) -> None:
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(
        json.dumps({"image_width": 100, "image_height": 100, "words": [], "lines": []}),
        encoding="utf-8",
    )
    gateway = FakeLlmGateway([json.dumps(_valid_receipt())])

    result = run_llm_main_parser(
        ocr_json_path=ocr_path,
        ollama_url="http://not-used",
        model="test-model",
        json_retry_count=0,
        llm_gateway=gateway,
    )

    assert result["error"] is None
    assert result["receipt"]["merchant"]["name"] == "TEST"
    assert len(gateway.requests) == 1
    assert gateway.requests[0].model == "test-model"
    assert gateway.requests[0].response_json_schema is not None
    assert gateway.requests[0].response_json_schema["properties"]["schema_version"]["enum"] == [
        "v14_6_llm_receipt_1"
    ]


def test_vlm_request_does_not_expose_backend_configuration() -> None:
    field_names = {field.name for field in fields(VlmRequest)}

    assert "backend_name" not in field_names
    assert "service_url" not in field_names
    assert "trusted_command" not in field_names


def test_ocr_request_uses_provider_neutral_option_names() -> None:
    field_names = {field.name for field in fields(OcrRequest)}

    assert "max_side_length" in field_names
    assert "detect_orientation" in field_names
    assert "detection_max_side_length" in field_names
    assert "use_angle_cls" not in field_names
    assert "det_limit_side_len" not in field_names
