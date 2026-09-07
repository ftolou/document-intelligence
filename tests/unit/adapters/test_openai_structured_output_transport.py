from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

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


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }


def test_generation_schema_uses_strict_output_without_prompt_duplication() -> None:
    client = _Client('{"value":7}')
    gateway = OpenAIGenerationGateway(client=client)

    gateway.generate(
        GenerationRequest(
            model="opaque-model",
            prompt="Do the task.",
            response_json_schema=_schema(),
        )
    )

    payload = client.responses.calls[-1]
    assert payload["input"] == "Do the task."
    assert payload["text"]["format"] == {
        "type": "json_schema",
        "name": "generation",
        "schema": _schema(),
        "strict": True,
    }


def test_chat_schema_uses_strict_output_without_prompt_duplication() -> None:
    client = _Client('{"value":7}')
    gateway = OpenAIChatGateway(client=client)

    gateway.generate(
        ChatGenerationRequest(
            model="opaque-model",
            system_prompt="System intent.",
            user_prompt="Do the task.",
            response_json_schema=_schema(),
        )
    )

    payload = client.responses.calls[-1]
    assert payload["instructions"] == "System intent."
    assert payload["input"] == "Do the task."
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True


def test_multimodal_schema_does_not_append_schema_to_model_input(tmp_path: Path) -> None:
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"png-data")
    client = _Client('{"value":7}')
    gateway = OpenAIMultimodalGateway(client=client)

    gateway.generate(
        MultimodalGenerationRequest(
            model="opaque-model",
            prompt="Read the image.",
            image_paths=(image_path,),
            format_json=True,
            response_json_schema=_schema(),
        )
    )

    payload = client.responses.calls[-1]
    content = payload["input"][0]["content"]
    assert [part["type"] for part in content] == ["input_text", "input_image"]
    assert content[0]["text"] == "Read the image."
    assert payload["text"]["format"]["type"] == "json_schema"


def test_incompatible_schema_fails_before_provider_call() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"metadata": {"type": "object"}},
        "required": ["metadata"],
    }
    client = _Client("{}")

    with pytest.raises(ValueError, match="not compatible"):
        OpenAIGenerationGateway(client=client).generate(
            GenerationRequest(
                model="opaque-model",
                prompt="Do the task.",
                response_json_schema=schema,
            )
        )

    assert client.responses.calls == []
