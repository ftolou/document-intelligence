from __future__ import annotations

import json

import pytest

from receipt_intelligence.application.ports.llm import GenerationRequest, GenerationResult
from receipt_intelligence.rag_sql.question_analyzer import (
    QuestionAnalysisError,
    QuestionAnalyzerConfig,
    RagSqlQuestionAnalyzer,
)


class FakeGateway:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(text=self.response)


def test_analyzer_extracts_product_entity_and_spending_goal() -> None:
    def generate(**kwargs: object) -> str:
        prompt = str(kwargs["prompt"])
        assert "Wie viel habe ich für Schuhe" in prompt
        assert "Never rewrite a request to show or list receipts" in prompt
        return json.dumps(
            {
                "schema_version": "rag_sql_question_analysis_v2",
                "status": "ready",
                "language": "de",
                "user_goal": "Berechne die gesamten Ausgaben für Schuhe.",
                "target_entity": "spending_amount",
                "requested_operation": "aggregate_sum",
                "requires_product_resolution": True,
                "entities": [
                    {
                        "entity_id": "e001",
                        "search_text": "Schuhe",
                        "role": "product_filter",
                    }
                ],
                "clarification_question": None,
                "reason": None,
            }
        )

    result = RagSqlQuestionAnalyzer(
        QuestionAnalyzerConfig(retry_count=0), generate=generate
    ).analyze("Wie viel habe ich für Schuhe ausgegeben?")

    assert result.status == "ready"
    assert result.entities[0].search_text == "Schuhe"
    assert result.target_entity == "spending_amount"
    assert result.requested_operation == "aggregate_sum"
    assert result.attempts == 1


def test_analyzer_preserves_receipt_lookup_instead_of_aggregation() -> None:
    def generate(**kwargs: object) -> str:
        prompt = str(kwargs["prompt"])
        assert "Zeige mir die Quittung mit Vittel" in prompt
        assert "target_entity=receipt and requested_operation=list" in prompt
        return json.dumps(
            {
                "schema_version": "rag_sql_question_analysis_v2",
                "status": "ready",
                "language": "de",
                "user_goal": "Zeige die Quittung oder Quittungen mit Vittel.",
                "target_entity": "receipt",
                "requested_operation": "list",
                "requires_product_resolution": True,
                "entities": [
                    {
                        "entity_id": "e001",
                        "search_text": "Vittel",
                        "role": "product_filter",
                    }
                ],
                "clarification_question": None,
                "reason": None,
            }
        )

    result = RagSqlQuestionAnalyzer(
        QuestionAnalyzerConfig(retry_count=0), generate=generate
    ).analyze("Zeige mir die Quittung mit Vittel.")

    assert result.user_goal == "Zeige die Quittung oder Quittungen mit Vittel."
    assert result.target_entity == "receipt"
    assert result.requested_operation == "list"
    assert result.entities[0].search_text == "Vittel"


def test_analyzer_allows_general_spending_without_product_rag() -> None:
    def generate(**_: object) -> str:
        return json.dumps(
            {
                "schema_version": "rag_sql_question_analysis_v2",
                "status": "ready",
                "language": "de",
                "user_goal": "Berechne die gesamten Ausgaben.",
                "target_entity": "spending_amount",
                "requested_operation": "aggregate_sum",
                "requires_product_resolution": False,
                "entities": [],
                "clarification_question": None,
                "reason": None,
            }
        )

    result = RagSqlQuestionAnalyzer(
        QuestionAnalyzerConfig(retry_count=0), generate=generate
    ).analyze("Wie viel habe ich insgesamt ausgegeben?")

    assert result.requires_product_resolution is False
    assert result.entities == []


def test_analyzer_retries_invalid_json_without_deterministic_fallback() -> None:
    calls = 0

    def generate(**_: object) -> str:
        nonlocal calls
        calls += 1
        return "not-json"

    analyzer = RagSqlQuestionAnalyzer(QuestionAnalyzerConfig(retry_count=1), generate=generate)
    with pytest.raises(QuestionAnalysisError, match="2 attempt"):
        analyzer.analyze("Schuhe")

    assert calls == 2


def test_analyzer_retains_ollama_timing_metrics() -> None:
    from receipt_intelligence.application.ports.llm import (
        GenerationResult,
        ModelCallMetrics,
    )

    metrics = ModelCallMetrics(
        provider="ollama",
        endpoint="generate",
        model="gemma4",
        request_duration_ms=250.0,
        load_duration_ns=100_000_000,
        prompt_eval_count=25,
        prompt_eval_duration_ns=50_000_000,
        eval_count=10,
        eval_duration_ns=80_000_000,
    )

    def generate(**_: object) -> str:
        return GenerationResult(
            text=json.dumps(
                {
                    "schema_version": "rag_sql_question_analysis_v2",
                    "status": "ready",
                    "language": "de",
                    "user_goal": "Berechne die gesamten Ausgaben.",
                    "target_entity": "spending_amount",
                    "requested_operation": "aggregate_sum",
                    "requires_product_resolution": False,
                    "entities": [],
                    "clarification_question": None,
                    "reason": None,
                }
            ),
            metrics=metrics,
        )

    result = RagSqlQuestionAnalyzer(
        QuestionAnalyzerConfig(retry_count=0), generate=generate
    ).analyze("Wie viel habe ich insgesamt ausgegeben?")

    assert result.ollama_calls == [metrics]


def test_analyzer_prompt_defines_descriptive_product_operations() -> None:
    def generate(**kwargs: object) -> str:
        prompt = str(kwargs["prompt"])
        assert '"What is Vittel?"' in prompt
        assert "requested_operation=describe_product" in prompt
        assert "requested_operation=identify_brand" in prompt
        assert "requested_operation=identify_product_type" in prompt
        return json.dumps(
            {
                "schema_version": "rag_sql_question_analysis_v2",
                "status": "ready",
                "language": "en",
                "user_goal": "Describe Vittel.",
                "target_entity": "product_description",
                "requested_operation": "describe_product",
                "requires_product_resolution": True,
                "entities": [
                    {"entity_id": "e001", "search_text": "Vittel", "role": "product_filter"}
                ],
                "clarification_question": None,
                "reason": None,
            }
        )

    result = RagSqlQuestionAnalyzer(
        QuestionAnalyzerConfig(retry_count=0), generate=generate
    ).analyze("What is Vittel?")
    assert result.requested_operation == "describe_product"


def test_analyzer_accepts_provider_neutral_gateway() -> None:
    gateway = FakeGateway(
        json.dumps(
            {
                "schema_version": "rag_sql_question_analysis_v2",
                "status": "ready",
                "language": "de",
                "user_goal": "Berechne die Ausgaben.",
                "target_entity": "spending_amount",
                "requested_operation": "aggregate_sum",
                "requires_product_resolution": False,
                "entities": [],
                "clarification_question": None,
                "reason": None,
            }
        )
    )

    result = RagSqlQuestionAnalyzer(
        QuestionAnalyzerConfig(retry_count=0), llm_gateway=gateway
    ).analyze("Wie viel habe ich ausgegeben?")

    assert result.status == "ready"
    assert len(gateway.requests) == 1
    assert gateway.requests[0].model == "gemma4"
