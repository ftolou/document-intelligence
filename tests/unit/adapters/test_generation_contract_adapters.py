from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from receipt_intelligence.adapters.llm import (
    OpenAIChatGateway,
    OpenAIGenerationGateway,
    OpenAIMultimodalGateway,
)
from receipt_intelligence.adapters.llm import ollama_gateway as ollama_module
from receipt_intelligence.application.llm_json import LLMJsonParseError, parse_json_from_llm
from receipt_intelligence.application.ports.chat import ChatGenerationRequest
from receipt_intelligence.application.ports.llm import (
    GenerationError,
    GenerationFailureReason,
    GenerationIncompleteError,
    GenerationProviderUnavailableError,
    GenerationRefusedError,
    GenerationRequest,
    GenerationResult,
    MalformedGenerationError,
)
from receipt_intelligence.application.ports.multimodal import MultimodalGenerationRequest


@dataclass
class _Response:
    output_text: str = '{"value": 7}'
    status: str = "completed"
    model: str = "opaque-model"
    output: list[dict[str, Any]] | None = None
    incomplete_details: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "output_text": self.output_text,
            "status": self.status,
            "model": self.model,
            "output": self.output or [],
            "incomplete_details": self.incomplete_details,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }


class _Responses:
    def __init__(self, response: _Response | Exception | None = None) -> None:
        self.response = response or _Response()
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _Client:
    def __init__(self, response: _Response | Exception | None = None) -> None:
        self.responses = _Responses(response)


def _request(**overrides: Any) -> GenerationRequest:
    values: dict[str, Any] = {"model": "opaque-model", "prompt": "Return JSON."}
    values.update(overrides)
    return GenerationRequest(**values)


def test_openai_omits_unset_options_and_preserves_explicit_zero() -> None:
    client = _Client()
    gateway = OpenAIGenerationGateway(client=client)

    gateway.generate(_request())
    omitted = client.responses.calls[-1]
    assert "temperature" not in omitted
    assert "reasoning" not in omitted

    gateway.generate(_request(temperature=0.0))
    assert client.responses.calls[-1]["temperature"] == 0.0


@pytest.mark.parametrize("think", [False, True])
def test_openai_reasoning_is_adapter_configuration_not_think(think: bool) -> None:
    client = _Client()
    gateway = OpenAIChatGateway(client=client, reasoning_effort="provider-new-value")

    gateway.generate(
        ChatGenerationRequest(
            model="opaque-model",
            system_prompt=None,
            user_prompt="Return JSON.",
            think=think,
        )
    )

    assert client.responses.calls[-1]["reasoning"] == {"effort": "provider-new-value"}


def test_model_identifier_is_forwarded_without_request_policy() -> None:
    client = _Client()
    gateway = OpenAIGenerationGateway(client=client, reasoning_effort="high")

    for model in ("future-model-alpha", "unrelated-model-2031"):
        gateway.generate(_request(model=model, temperature=0.25))

    first, second = client.responses.calls
    assert first["model"] == "future-model-alpha"
    assert second["model"] == "unrelated-model-2031"
    assert {key for key in first if key != "model"} == {key for key in second if key != "model"}
    assert first["reasoning"] == second["reasoning"] == {"effort": "high"}
    assert first["temperature"] == second["temperature"] == 0.25


def test_openai_multimodal_translation_is_confined_to_adapter(tmp_path: Path) -> None:
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"png-data")
    client = _Client()
    gateway = OpenAIMultimodalGateway(
        client=client,
        reasoning_effort="high",
        image_detail="high",
    )

    gateway.generate(
        MultimodalGenerationRequest(
            model="opaque-model",
            system_prompt="System intent.",
            prompt="Read the image.",
            image_paths=(image_path,),
        )
    )

    payload = client.responses.calls[-1]
    assert payload["instructions"] == "System intent."
    image = payload["input"][0]["content"][1]
    assert image["type"] == "input_image"
    assert image["detail"] == "high"
    assert image["image_url"].startswith("data:image/png;base64,")
    assert payload["reasoning"] == {"effort": "high"}
    assert "temperature" not in payload


def test_structured_transport_uses_strict_schema_without_mutating_original() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "integer", "minimum": 0}},
        "required": ["value"],
    }
    original = copy.deepcopy(schema)
    client = _Client()
    result = OpenAIGenerationGateway(client=client).generate(_request(response_json_schema=schema))

    transport = client.responses.calls[-1]["text"]["format"]
    assert transport["type"] == "json_schema"
    assert transport["strict"] is True
    assert "minimum" not in transport["schema"]["properties"]["value"]
    assert schema == original
    assert parse_json_from_llm(result, response_json_schema=schema) == {"value": 7}


@pytest.mark.parametrize(
    "dynamic_property",
    [
        {"type": "object"},
        {"type": ["object", "null"]},
        {"type": "object", "additionalProperties": {"type": "string"}},
    ],
)
def test_dynamic_and_nullable_objects_fall_back_without_schema_narrowing(
    dynamic_property: dict[str, Any],
) -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"metadata": dynamic_property},
        "required": ["metadata"],
    }
    client = _Client(_Response(output_text='{"metadata":{"extra":"ok"}}'))
    result = OpenAIGenerationGateway(client=client).generate(_request(response_json_schema=schema))

    assert client.responses.calls[-1]["text"] == {"format": {"type": "json_object"}}
    assert parse_json_from_llm(result, response_json_schema=schema)["metadata"] == {"extra": "ok"}


def test_optional_properties_fall_back_instead_of_becoming_required() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"optional_value": {"type": "string"}},
    }
    client = _Client(_Response(output_text="{}"))
    result = OpenAIGenerationGateway(client=client).generate(_request(response_json_schema=schema))

    assert client.responses.calls[-1]["text"] == {"format": {"type": "json_object"}}
    assert parse_json_from_llm(result, response_json_schema=schema) == {}


@pytest.mark.parametrize(
    ("keyword", "constraint"),
    [
        ("allOf", [{"required": ["left"]}]),
        ("not", {"required": ["other"]}),
        ("dependentRequired", {"left": ["right"]}),
        ("dependentSchemas", {"left": {"required": ["right"]}}),
        ("if", {"required": ["left"]}),
        ("then", {"required": ["right"]}),
        ("else", {"required": ["right"]}),
    ],
)
def test_unsupported_strict_compositions_fall_back_without_schema_narrowing(
    keyword: str,
    constraint: Any,
) -> None:
    value_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"left": {"type": "string"}, "right": {"type": "string"}},
        "required": ["left", "right"],
        keyword: constraint,
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": value_schema},
        "required": ["value"],
    }
    output = {"value": {"left": "a", "right": "b"}}
    client = _Client(_Response(output_text=json.dumps(output)))
    result = OpenAIGenerationGateway(client=client).generate(_request(response_json_schema=schema))

    assert client.responses.calls[-1]["text"] == {"format": {"type": "json_object"}}
    assert parse_json_from_llm(result, response_json_schema=schema) == output


@pytest.mark.parametrize(
    ("keyword", "constraint"),
    [
        ("oneOf", [{"type": "string"}, {"type": "integer"}]),
        ("prefixItems", [{"type": "string"}]),
        ("contains", {"type": "string"}),
        ("propertyNames", {"pattern": "^[a-z]+$"}),
    ],
)
def test_other_unsupported_strict_keywords_fall_back_to_json_mode(
    keyword: str,
    constraint: Any,
) -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "value": {
                "type": "string",
                keyword: constraint,
            }
        },
        "required": ["value"],
    }
    client = _Client(_Response(output_text='{"value":"ok"}'))
    result = OpenAIGenerationGateway(client=client).generate(
        _request(response_json_schema=schema)
    )

    assert client.responses.calls[-1]["text"] == {"format": {"type": "json_object"}}
    assert parse_json_from_llm(result, response_json_schema=schema) == {"value": "ok"}


@pytest.mark.parametrize(
    ("schema", "output"),
    [
        (
            {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                    },
                ]
            },
            {"value": "ok"},
        ),
        (
            {
                "additionalProperties": False,
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            {"value": "ok"},
        ),
    ],
)
def test_unsupported_strict_root_shapes_fall_back_without_schema_narrowing(
    schema: dict[str, Any],
    output: Any,
) -> None:
    client = _Client(_Response(output_text=json.dumps(output)))
    result = OpenAIGenerationGateway(client=client).generate(_request(response_json_schema=schema))

    assert client.responses.calls[-1]["text"] == {"format": {"type": "json_object"}}
    assert parse_json_from_llm(result, response_json_schema=schema) == output


@pytest.mark.parametrize("keyword", ["const", "enum"])
def test_strict_schema_preserves_object_literals(keyword: str) -> None:
    literal = {
        "required": "kept",
        "minimum": 3,
        "default": "kept",
        "additionalProperties": True,
        "allOf": "literal",
    }
    literal_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "required": {"type": "string"},
            "minimum": {"type": "integer"},
            "default": {"type": "string"},
            "additionalProperties": {"type": "boolean"},
            "allOf": {"type": "string"},
        },
        "required": list(literal),
        keyword: literal if keyword == "const" else [literal],
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": literal_schema},
        "required": ["value"],
    }
    original = copy.deepcopy(schema)
    output = {"value": literal}
    client = _Client(_Response(output_text=json.dumps(output)))
    result = OpenAIGenerationGateway(client=client).generate(_request(response_json_schema=schema))

    transport = client.responses.calls[-1]["text"]["format"]
    assert transport["type"] == "json_schema"
    expected_constraint = literal if keyword == "const" else [literal]
    assert transport["schema"]["properties"]["value"][keyword] == expected_constraint
    assert schema == original
    assert parse_json_from_llm(result, response_json_schema=schema) == output


def test_original_schema_rejects_output_accepted_by_broader_transport() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "integer", "minimum": 10}},
        "required": ["value"],
    }
    client = _Client(_Response(output_text='{"value":7}'))
    result = OpenAIGenerationGateway(client=client).generate(_request(response_json_schema=schema))

    with pytest.raises(LLMJsonParseError, match="does not match"):
        parse_json_from_llm(result, response_json_schema=schema)


@pytest.mark.parametrize(
    ("response", "error_type", "reason"),
    [
        (
            _Response(status="incomplete", incomplete_details={"reason": "max_tokens"}),
            GenerationIncompleteError,
            GenerationFailureReason.INCOMPLETE,
        ),
        (
            _Response(output_text="", output=[{"type": "refusal", "refusal": "no"}]),
            GenerationRefusedError,
            GenerationFailureReason.REFUSED,
        ),
        (
            _Response(output_text=""),
            MalformedGenerationError,
            GenerationFailureReason.MALFORMED_OUTPUT,
        ),
    ],
)
def test_openai_response_failures_are_normalized(
    response: _Response,
    error_type: type[GenerationError],
    reason: GenerationFailureReason,
) -> None:
    with pytest.raises(error_type) as captured:
        OpenAIGenerationGateway(client=_Client(response)).generate(_request())
    assert captured.value.reason is reason
    assert captured.value.provider == "openai"


def test_openai_transport_failures_are_normalized() -> None:
    provider_error = RuntimeError("bad option")
    provider_error.status_code = 400  # type: ignore[attr-defined]
    unavailable = RuntimeError("service down")
    unavailable.status_code = 503  # type: ignore[attr-defined]

    with pytest.raises(GenerationError) as bad_request:
        OpenAIGenerationGateway(client=_Client(provider_error)).generate(_request())
    assert type(bad_request.value) is GenerationError
    assert bad_request.value.reason is GenerationFailureReason.PROVIDER_ERROR

    with pytest.raises(GenerationProviderUnavailableError) as service_down:
        OpenAIGenerationGateway(client=_Client(unavailable)).generate(_request())
    assert service_down.value.reason is GenerationFailureReason.PROVIDER_UNAVAILABLE


def test_ollama_temperature_omission_and_explicit_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, Any]] = []

    def fake_http_json(
        url: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        del timeout
        if url.endswith("/api/tags"):
            return {"models": []}
        payloads.append(dict(payload or {}))
        return {"response": "ok", "done": True, "model": "opaque-model"}

    monkeypatch.setattr(ollama_module, "_http_json", fake_http_json)
    gateway = ollama_module.OllamaGateway("http://ollama.invalid")

    gateway.generate(_request())
    gateway.generate(_request(temperature=0.0))

    assert "temperature" not in payloads[0]["options"]
    assert payloads[1]["options"]["temperature"] == 0.0


def test_generation_adapter_contains_no_model_capability_policy() -> None:
    source = Path(OpenAIGenerationGateway.__module__.replace(".", "/") + ".py")
    repository_source = Path("src") / source
    text = repository_source.read_text(encoding="utf-8")
    assert "model.startswith(" not in text
    assert "model_id.startswith(" not in text
    assert "supports_temperature(" not in text
    assert "reasoning_effort_for_request(" not in text
