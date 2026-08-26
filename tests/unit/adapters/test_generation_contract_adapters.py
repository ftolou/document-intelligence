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
    assert request["timeout"] == 120.0


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
            model="provider-model",
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
    image = request["input"][0]["content"][1]
    assert image["image_url"].startswith("data:image/png;base64,")
    assert image["detail"] == "high"


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
