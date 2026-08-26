from __future__ import annotations

import json

import pytest

from receipt_intelligence.application.ports.llm import GenerationRequest, GenerationResult
from receipt_intelligence.rag_sql.answer_formatter import (
    AnswerFormatterConfig,
    AnswerFormatterResult,
    AnswerFormattingError,
    EvidenceBoundAnswerFormatter,
    render_validated_answer,
    validate_answer_formatter_result,
)


class FakeGateway:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(text=json.dumps(self.payload))


def _rows() -> list[dict[str, object]]:
    return [
        {
            "item_id": 175,
            "description": "STARBUCKS NESPRESSO CAPSULES",
            "normalized_name": "Starbucks Nespresso Capsules",
            "semantic_description": None,
            "category": "groceries_food",
            "category_reason": (
                "The packaging identifies Starbucks as the product brand and "
                "Nespresso as the compatible system."
            ),
        }
    ]


def _formatter(payload: dict[str, object]) -> EvidenceBoundAnswerFormatter:
    def generate(**_kwargs: object) -> str:
        return json.dumps(payload)

    return EvidenceBoundAnswerFormatter(
        AnswerFormatterConfig(model="test-model", retry_count=0),
        generate=generate,
    )


def test_formatter_returns_structured_evidence_only() -> None:
    gateway = FakeGateway(
        {
            "schema_version": "rag_sql_answer_format_v1",
            "status": "resolved",
            "values": ["Starbucks"],
            "supporting_item_ids": [175],
            "evidence_fields": ["description", "category_reason"],
            "reason": "Starbucks is explicitly identified as the product brand.",
        }
    )
    formatter = EvidenceBoundAnswerFormatter(
        AnswerFormatterConfig(model="test-model", retry_count=0, format_json=False),
        llm_gateway=gateway,
    )

    result = formatter.format(
        question="What was the coffee brand?",
        requested_operation="identify_brand",
        language="en",
        rows=_rows(),
        answer_instruction="Identify the reviewed product brand.",
    )

    assert result.status == "resolved"
    assert result.values == ["Starbucks"]
    assert result.model == "test-model"
    assert gateway.requests[0].format_json is False
    assert gateway.requests[0].response_json_schema is None


def test_validator_accepts_supported_brand_and_renders_deterministically() -> None:
    result = AnswerFormatterResult(
        status="resolved",
        values=["Starbucks"],
        supporting_item_ids=[175],
        evidence_fields=["description", "category_reason"],
        reason="Explicit reviewed product branding.",
        model="test-model",
        attempts=1,
    )

    validation = validate_answer_formatter_result(
        result,
        requested_operation="identify_brand",
        rows=_rows(),
    )

    assert validation.status == "valid"
    assert (
        render_validated_answer(
            validation,
            requested_operation="identify_brand",
            language="en",
        )
        == "The brand named in the reviewed product data is “Starbucks”."
    )


def test_validator_rejects_unknown_supporting_item_id() -> None:
    result = AnswerFormatterResult(
        status="resolved",
        values=["Starbucks"],
        supporting_item_ids=[999],
        evidence_fields=["description"],
        reason="Claimed evidence.",
    )

    validation = validate_answer_formatter_result(
        result,
        requested_operation="identify_brand",
        rows=_rows(),
    )

    assert validation.status == "invalid"
    assert "unknown_supporting_item_ids" in validation.reason


def test_validator_rejects_value_absent_from_reviewed_rows() -> None:
    result = AnswerFormatterResult(
        status="resolved",
        values=["Jacobs"],
        supporting_item_ids=[175],
        evidence_fields=["description", "category_reason"],
        reason="Unsupported claim.",
    )

    validation = validate_answer_formatter_result(
        result,
        requested_operation="identify_brand",
        rows=_rows(),
    )

    assert validation.status == "invalid"
    assert validation.reason == "unsupported_value:Jacobs"


def test_validator_rejects_compatible_system_as_brand() -> None:
    result = AnswerFormatterResult(
        status="resolved",
        values=["Nespresso"],
        supporting_item_ids=[175],
        evidence_fields=["description", "category_reason"],
        reason="Nespresso appears in the row.",
    )

    validation = validate_answer_formatter_result(
        result,
        requested_operation="identify_brand",
        rows=_rows(),
    )
    assert validation.status == "invalid"
    assert validation.reason == "brand_assigned_non_brand_role:Nespresso"


def test_validator_rejects_merchant_only_brand_guess() -> None:
    rows = [
        {
            "item_id": 5,
            "description": "CLASSIC COFFEE PADS",
            "normalized_name": "Classic Coffee Pads",
            "category_reason": "Coffee pads sold by REWE.",
            "merchant": "REWE",
        }
    ]
    result = AnswerFormatterResult(
        status="resolved",
        values=["REWE"],
        supporting_item_ids=[5],
        evidence_fields=["category_reason"],
        reason="Invalid merchant inference.",
    )

    validation = validate_answer_formatter_result(
        result,
        requested_operation="identify_brand",
        rows=rows,
    )

    assert validation.status == "invalid"
    assert "brand_not_present_in_product_identity" in validation.reason


def test_formatter_retries_malformed_json_then_fails() -> None:
    calls = 0

    def generate(**_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return "not-json"

    formatter = EvidenceBoundAnswerFormatter(
        AnswerFormatterConfig(model="test-model", retry_count=1),
        generate=generate,
    )

    with pytest.raises(AnswerFormattingError):
        formatter.format(
            question="What was the coffee brand?",
            requested_operation="identify_brand",
            language="en",
            rows=_rows(),
            answer_instruction="Identify the brand.",
        )
    assert calls == 2
