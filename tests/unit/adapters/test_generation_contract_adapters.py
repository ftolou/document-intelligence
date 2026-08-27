from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from receipt_intelligence.adapters.chat.ollama import OllamaChatGateway
from receipt_intelligence.adapters.llm.openai_responses import (
    OpenAIChatGateway,
    OpenAIGenerationGateway,
    OpenAIMultimodalGateway,
)
from receipt_intelligence.adapters.multimodal.ollama import OllamaMultimodalGateway
from receipt_intelligence.application.llm_json import LLMJsonParseError, parse_json_from_llm
from receipt_intelligence.application.ports.chat import ChatGenerationRequest
from receipt_intelligence.application.ports.llm import (
    GenerationError,
    GenerationFailureReason,
    GenerationIncompleteError,
    GenerationProviderUnavailableError,
    GenerationRefusedError,
    GenerationRequest,
)
from receipt_intelligence.application.ports.multimodal import MultimodalGenerationRequest
from receipt_intelligence.rag_sql.models import (
    QuestionAnalysisResult,
    ResolvedSemanticEntity,
    SemanticEntity,
)
from receipt_intelligence.rag_sql.planner import (
    RagSqlPlanner,
    RagSqlPlannerConfig,
    build_protected_item_parameters,
)
from receipt_intelligence.rag_sql.question_analyzer import (
    QuestionAnalyzerConfig,
    RagSqlQuestionAnalyzer,
)


class FakeResponses:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def _response(text: str, **overrides: Any) -> SimpleNamespace:
    values = {
        "id": "resp_1",
        "model": "provider-model",
        "status": "completed",
        "output_text": text,
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _assert_openai_strict_schema(schema: Any) -> None:
    if isinstance(schema, list):
        for item in schema:
            _assert_openai_strict_schema(item)
        return
    if not isinstance(schema, dict):
        return

    unsupported = {
        "$schema",
        "default",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "pattern",
        "uniqueItems",
    }
    assert unsupported.isdisjoint(schema)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        assert schema["required"] == list(properties)
        assert schema["additionalProperties"] is False
    for value in schema.values():
        _assert_openai_strict_schema(value)


def test_openai_text_adapter_runs_existing_rag_workflow_with_structured_schema() -> None:
    payload = {
        "schema_version": "rag_sql_question_analysis_v2",
        "status": "ready",
        "language": "de",
        "user_goal": "Berechne Ausgaben fuer Schuhe.",
        "target_entity": "spending_amount",
        "requested_operation": "aggregate_sum",
        "requires_product_resolution": True,
        "entities": [{"entity_id": "e001", "search_text": "Schuhe", "role": "product_filter"}],
        "clarification_question": None,
        "reason": None,
    }
    responses = FakeResponses(_response(json.dumps(payload)))
    gateway = OpenAIGenerationGateway(client=FakeClient(responses))

    result = RagSqlQuestionAnalyzer(
        QuestionAnalyzerConfig(retry_count=0),
        llm_gateway=gateway,
    ).analyze("Wie viel habe ich fuer Schuhe ausgegeben?")

    assert result.target_entity == "spending_amount"
    request = responses.calls[0]
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    schema = request["text"]["format"]["schema"]
    _assert_openai_strict_schema(schema)
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["language"]["type"] == ["string", "null"]
    assert request["temperature"] == 0.0
    assert request["timeout"] == 120.0
    assert "reasoning" not in request


def test_openai_text_adapter_falls_back_for_planner_dynamic_parameters() -> None:
    resolved = [
        ResolvedSemanticEntity(
            entity_id="e001",
            search_text="shoes",
            status="resolved",
            selected_item_ids=[84, 126],
        )
    ]
    protected = build_protected_item_parameters(resolved)
    payload = {
        "schema_version": "rag_sql_plan_v2",
        "status": "ready",
        "sql": (
            "SELECT SUM(line_total) AS value FROM analytics_purchase_items "
            "WHERE item_id IN (:e001_item_0, :e001_item_1)"
        ),
        "parameters": protected,
        "result_shape": "scalar",
        "result_entity": "spending_amount",
        "display_columns": ["value"],
        "answer_instruction": "Report spending on the resolved items.",
        "clarification_question": None,
        "reason": None,
    }
    responses = FakeResponses(_response(json.dumps(payload)))
    gateway = OpenAIGenerationGateway(client=FakeClient(responses))

    result = RagSqlPlanner(
        RagSqlPlannerConfig(retry_count=0),
        llm_gateway=gateway,
    ).plan(
        "How much did I spend on shoes?",
        analysis=QuestionAnalysisResult(
            status="ready",
            language="en",
            user_goal="Calculate spending on shoes.",
            target_entity="spending_amount",
            requested_operation="aggregate_sum",
            requires_product_resolution=True,
            entities=[SemanticEntity(entity_id="e001", search_text="shoes")],
        ),
        resolved_entities=resolved,
        protected_parameters=protected,
    )

    assert result.parameters == {"e001_item_0": 84, "e001_item_1": 126}
    assert responses.calls[0]["text"]["format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object"},
        {"type": "object", "properties": {"known": {"type": "string"}}},
        {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"known": {"type": "string"}},
                }
            },
            "required": ["nested"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"metadata": {"type": ["object", "null"]}},
            "required": ["metadata"],
            "additionalProperties": False,
        },
    ],
)
def test_openai_text_adapter_falls_back_for_implicitly_open_objects(
    schema: dict[str, Any],
) -> None:
    responses = FakeResponses(_response('{"known":"value"}'))
    gateway = OpenAIGenerationGateway(client=FakeClient(responses))

    gateway.generate(
        GenerationRequest(
            model="gpt-4.1",
            prompt="return json",
            response_json_schema=schema,
        )
    )

    assert responses.calls[0]["text"]["format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_reasoning"),
    [
        (None, None),
        ("none", {"effort": "none"}),
    ],
)
def test_openai_generation_adapter_distinguishes_unset_and_explicit_none(
    reasoning_effort: str | None,
    expected_reasoning: dict[str, str] | None,
) -> None:
    responses = FakeResponses(_response("done"))
    gateway = OpenAIGenerationGateway(
        client=FakeClient(responses),
        reasoning_effort=reasoning_effort,
    )

    gateway.generate(GenerationRequest(model="gpt-5.6-luna", prompt="answer"))

    request = responses.calls[0]
    if expected_reasoning is None:
        assert "reasoning" not in request
    else:
        assert request["reasoning"] == expected_reasoning


def test_openai_chat_adapter_normalizes_refusal() -> None:
    responses = FakeResponses(
        _response(
            "",
            output=[{"content": [{"type": "refusal", "refusal": "cannot comply"}]}],
        )
    )
    gateway = OpenAIChatGateway(client=FakeClient(responses))

    with pytest.raises(GenerationRefusedError) as raised:
        gateway.generate(
            ChatGenerationRequest(
                model="provider-model",
                system_prompt="system",
                user_prompt="user",
            )
        )

    assert raised.value.reason is GenerationFailureReason.REFUSED
    assert raised.value.provider == "openai"
    assert "reasoning" not in responses.calls[0]


def test_openai_multimodal_adapter_preserves_separate_image_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"image")
    responses = FakeResponses(_response('{"ok":true}'))
    gateway = OpenAIMultimodalGateway(
        client=FakeClient(responses),
        reasoning_effort="medium",
        image_detail="high",
    )

    result = gateway.generate(
        MultimodalGenerationRequest(
            model="o3",
            system_prompt="extract carefully",
            prompt="extract receipt",
            image_paths=(image_path,),
            think=True,
            format_json=True,
            response_json_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
    )

    assert result.text == '{"ok":true}'
    request = responses.calls[0]
    assert request["instructions"] == "extract carefully"
    assert request["reasoning"] == {"effort": "medium"}
    assert "temperature" not in request
    image = request["input"][0]["content"][1]
    assert image["image_url"].startswith("data:image/png;base64,")
    assert image["detail"] == "high"


@pytest.mark.parametrize(
    ("model", "reasoning_effort", "expected_reasoning"),
    [
        ("gpt-5.6-luna", None, None),
        ("gpt-5.6-luna", "none", {"effort": "none"}),
        ("gpt-5.6-luna", "medium", {"effort": "none"}),
        ("gpt-5", "medium", None),
        ("gpt-5-pro", "high", None),
        ("o3", "medium", None),
        ("o4-mini", "low", None),
        ("provider-model", "medium", None),
        ("gpt-4.1", "medium", None),
    ],
)
def test_openai_multimodal_adapter_translates_think_false_to_no_reasoning(
    tmp_path: Path,
    model: str,
    reasoning_effort: str | None,
    expected_reasoning: dict[str, str] | None,
) -> None:
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"image")
    responses = FakeResponses(_response("done"))
    gateway = OpenAIMultimodalGateway(
        client=FakeClient(responses),
        reasoning_effort=reasoning_effort,
    )

    gateway.generate(
        MultimodalGenerationRequest(
            model=model,
            system_prompt=None,
            prompt="extract receipt",
            image_paths=(image_path,),
            think=False,
        )
    )

    request = responses.calls[0]
    if expected_reasoning is None:
        assert "reasoning" not in request
    else:
        assert request["reasoning"] == expected_reasoning


@pytest.mark.parametrize(
    ("model", "reasoning_effort", "expected_reasoning", "has_temperature"),
    [
        ("gpt-5", "minimal", {"effort": "minimal"}, False),
        ("gpt-5-pro", "high", {"effort": "high"}, False),
        ("gpt-5-pro", "medium", None, False),
        ("gpt-5.1", "none", {"effort": "none"}, True),
        ("gpt-5.1", "medium", {"effort": "medium"}, False),
        ("gpt-5.2", "none", {"effort": "none"}, True),
        ("o3", "none", None, False),
        ("o3", "medium", {"effort": "medium"}, False),
        ("o4-mini", "low", {"effort": "low"}, False),
        ("gpt-5.6-luna", "minimal", None, False),
        ("gpt-4.1", "medium", None, True),
        ("provider-model", "medium", None, True),
    ],
)
def test_openai_chat_adapter_filters_model_specific_request_parameters(
    model: str,
    reasoning_effort: str,
    expected_reasoning: dict[str, str] | None,
    has_temperature: bool,
) -> None:
    responses = FakeResponses(_response("done"))
    gateway = OpenAIChatGateway(
        client=FakeClient(responses),
        reasoning_effort=reasoning_effort,
    )

    gateway.generate(
        ChatGenerationRequest(
            model=model,
            system_prompt=None,
            user_prompt="answer",
            think=True,
        )
    )

    request = responses.calls[0]
    if expected_reasoning is None:
        assert "reasoning" not in request
    else:
        assert request["reasoning"] == expected_reasoning
    assert ("temperature" in request) is has_temperature


def test_ollama_multimodal_adapter_honors_deprecated_request_options(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"image")
    captured: dict[str, Any] = {}

    def fake_http_json(
        _url: str,
        *,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        captured.update(payload)
        assert timeout == 300.0
        return {"done": True, "message": {"content": "receipt text"}}

    gateway = OllamaMultimodalGateway(
        "http://ollama.test",
        generation_options={"seed": 7},
    )
    with pytest.deprecated_call(match="provider_options"):
        request = MultimodalGenerationRequest(
            model="local-model",
            prompt="extract receipt",
            image_paths=(image_path,),
            provider_options={"top_k": 20},
        )
    with patch(
        "receipt_intelligence.adapters.multimodal.ollama._http_json",
        side_effect=fake_http_json,
    ):
        result = gateway.generate(request)

    assert result.text == "receipt text"
    assert captured["options"]["seed"] == 7
    assert captured["options"]["top_k"] == 20


@pytest.mark.parametrize(
    ("response", "error_type", "reason"),
    [
        (
            _response("", status="incomplete", incomplete_details={"reason": "max_tokens"}),
            GenerationIncompleteError,
            GenerationFailureReason.INCOMPLETE,
        ),
        (
            None,
            GenerationProviderUnavailableError,
            GenerationFailureReason.PROVIDER_UNAVAILABLE,
        ),
    ],
)
def test_openai_adapter_normalizes_incomplete_and_unavailable(
    response: Any,
    error_type: type[GenerationError],
    reason: GenerationFailureReason,
) -> None:
    responses = (
        FakeResponses(response=response)
        if response is not None
        else FakeResponses(error=TimeoutError("timed out"))
    )
    gateway = OpenAIGenerationGateway(client=FakeClient(responses))

    with pytest.raises(error_type) as raised:
        gateway.generate(GenerationRequest(model="provider-model", prompt="hello"))

    assert raised.value.reason is reason


def test_ollama_adapter_uses_same_incomplete_semantics() -> None:
    with patch(
        "receipt_intelligence.adapters.chat.ollama._http_json",
        return_value={"done": False, "message": {"content": "partial"}},
    ):
        with pytest.raises(GenerationIncompleteError) as raised:
            OllamaChatGateway("http://ollama").generate(
                ChatGenerationRequest(
                    model="local-model",
                    system_prompt=None,
                    user_prompt="hello",
                )
            )

    assert raised.value.reason is GenerationFailureReason.INCOMPLETE
    assert raised.value.provider == "ollama"


def test_structured_parser_normalizes_schema_mismatch() -> None:
    schema = {
        "type": "object",
        "properties": {"status": {"const": "ok"}},
        "required": ["status"],
        "additionalProperties": False,
    }

    with pytest.raises(LLMJsonParseError) as raised:
        parse_json_from_llm('{"status":"wrong"}', response_json_schema=schema)

    assert raised.value.reason is GenerationFailureReason.MALFORMED_OUTPUT
