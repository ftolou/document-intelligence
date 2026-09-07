from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from receipt_intelligence.adapters.llm import (
    OpenAIChatGateway,
    OpenAIGenerationGateway,
    OpenAIMultimodalGateway,
)
from receipt_intelligence.application.ports.chat import ChatGenerationRequest
from receipt_intelligence.application.ports.llm import GenerationRequest
from receipt_intelligence.application.ports.multimodal import MultimodalGenerationRequest


@dataclass
class _Response:
    output_text: str
    status: str = "completed"
    model: str = "opaque-model"

    def model_dump(self) -> dict[str, Any]:
        return {
            "output_text": self.output_text,
            "status": self.status,
            "model": self.model,
            "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }


class _Responses:
    def __init__(self, output_text: str) -> None:
        self.response = _Response(output_text)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return self.response


class _Client:
    def __init__(self, output_text: str) -> None:
        self.responses = _Responses(output_text)


def _fallback_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"optional_value": {"type": "string"}},
    }


def _assert_schema_instruction(text: str) -> None:
    assert "JSON Schema" in text
    assert "<response_json_schema>" in text
    assert '"optional_value"' in text


def test_generation_fallback_puts_original_schema_in_model_input() -> None:
    client = _Client("{}")
    gateway = OpenAIGenerationGateway(client=client)

    gateway.generate(
        GenerationRequest(
            model="opaque-model",
            prompt="Do the task.",
            response_json_schema=_fallback_schema(),
        )
    )

    payload = client.responses.calls[-1]
    assert payload["text"] == {"format": {"type": "json_object"}}
    _assert_schema_instruction(payload["input"])


def test_chat_fallback_puts_original_schema_in_model_input() -> None:
    client = _Client("{}")
    gateway = OpenAIChatGateway(client=client)

    gateway.generate(
        ChatGenerationRequest(
            model="opaque-model",
            system_prompt="System intent.",
            user_prompt="Do the task.",
            response_json_schema=_fallback_schema(),
        )
    )

    payload = client.responses.calls[-1]
    assert payload["instructions"] == "System intent."
    assert payload["text"] == {"format": {"type": "json_object"}}
    _assert_schema_instruction(payload["input"])


def test_multimodal_fallback_appends_schema_without_replacing_image_input(tmp_path: Path) -> None:
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"png-data")
    client = _Client("{}")
    gateway = OpenAIMultimodalGateway(client=client)

    gateway.generate(
        MultimodalGenerationRequest(
            model="opaque-model",
            prompt="Read the image.",
            image_paths=(image_path,),
            format_json=True,
            response_json_schema=_fallback_schema(),
        )
    )

    payload = client.responses.calls[-1]
    content = payload["input"][0]["content"]
    assert payload["text"] == {"format": {"type": "json_object"}}
    assert [part["type"] for part in content] == ["input_text", "input_image", "input_text"]
    assert content[0]["text"] == "Read the image."
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    _assert_schema_instruction(content[2]["text"])


def test_strict_schema_transport_does_not_duplicate_schema_into_input() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    client = _Client('{"value":7}')
    gateway = OpenAIGenerationGateway(client=client)

    gateway.generate(
        GenerationRequest(
            model="opaque-model",
            prompt="Do the task.",
            response_json_schema=schema,
        )
    )

    payload = client.responses.calls[-1]
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["input"] == "Do the task."
