"""OpenAI Responses implementations of the provider-neutral generation ports."""

from __future__ import annotations

import base64
import io
import mimetypes
import re
import time
from pathlib import Path
from typing import Any

from receipt_intelligence.application.ports.chat import (
    ChatGenerationRequest,
    ChatGenerationResult,
)
from receipt_intelligence.application.ports.llm import (
    GenerationError,
    GenerationIncompleteError,
    GenerationProviderUnavailableError,
    GenerationRefusedError,
    GenerationRequest,
    GenerationResult,
    MalformedGenerationError,
    ModelCallMetrics,
)
from receipt_intelligence.application.ports.multimodal import (
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_SCHEMA_NAME = re.compile(r"[^a-zA-Z0-9_-]+")
_ORIGINAL_GPT_5_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}
_LATER_GPT_5_REASONING_EFFORTS = {"none", "low", "medium", "high"}
_O_SERIES_REASONING_EFFORTS = {"low", "medium", "high"}
_UNSUPPORTED_STRICT_SCHEMA_KEYWORDS = {
    "$schema",
    "default",
    "examples",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minItems",
    "minLength",
    "minProperties",
    "minimum",
    "multipleOf",
    "pattern",
    "uniqueItems",
}


class _OpenAIResponsesAdapter:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = 180.0,
        reasoning_effort: str | None = None,
        image_detail: str = "auto",
        client: Any | None = None,
    ) -> None:
        normalized_url = str(base_url or "").strip().rstrip("/")
        normalized_effort = str(reasoning_effort or "").strip().lower() or None
        normalized_detail = str(image_detail or "").strip().lower()
        if not normalized_url:
            raise ValueError("base_url must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if normalized_detail not in {"auto", "low", "high"}:
            raise ValueError("image_detail must be auto, low, or high.")
        if normalized_effort not in {None, "none", "minimal", "low", "medium", "high"}:
            raise ValueError("reasoning_effort is not supported.")

        self.reasoning_effort = normalized_effort
        self.image_detail = normalized_detail
        if client is not None:
            self._client = client
            return

        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise GenerationProviderUnavailableError(
                "The OpenAI adapter requires the optional 'openai' package.",
                provider="openai",
            ) from exc

        client_options: dict[str, Any] = {
            "base_url": normalized_url,
            "timeout": float(timeout_seconds),
        }
        normalized_key = str(api_key or "").strip()
        if normalized_key:
            client_options["api_key"] = normalized_key
        self._client = OpenAI(**client_options)

    def _generate(
        self,
        *,
        model: str,
        operation: str,
        timeout_seconds: float,
        num_predict: int,
        temperature: float,
        think: bool,
        input_value: Any,
        system_prompt: str | None,
        response_json_schema: dict[str, Any] | None,
        format_json: bool,
    ) -> tuple[str, dict[str, Any], ModelCallMetrics]:
        payload: dict[str, Any] = {
            "model": model,
            "store": False,
            "max_output_tokens": num_predict,
            "input": input_value,
        }
        if system_prompt:
            payload["instructions"] = system_prompt
        if response_json_schema is not None:
            if _has_dynamic_object_shape(response_json_schema):
                payload["text"] = {"format": {"type": "json_object"}}
            else:
                payload["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": _schema_name(operation),
                        "schema": _openai_strict_schema(response_json_schema),
                        "strict": True,
                    }
                }
        elif format_json:
            payload["text"] = {"format": {"type": "json_object"}}
        reasoning_effort = _reasoning_effort_for_request(
            model=model,
            configured_effort=self.reasoning_effort,
            think=think,
        )
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        if _supports_temperature(model=model, reasoning_effort=reasoning_effort):
            payload["temperature"] = temperature

        started = time.perf_counter()
        try:
            response = self._client.responses.create(
                **payload,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise _normalize_openai_error(exc) from exc
        duration_ms = (time.perf_counter() - started) * 1000.0
        raw_response = _response_to_dict(response)
        text = _response_text(response, raw_response)
        return text, raw_response, _metrics(response, model=model, duration_ms=duration_ms)


class OpenAIGenerationGateway(_OpenAIResponsesAdapter):
    """Text generation adapter usable by RAG and extraction classification."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        text, _raw, metrics = self._generate(
            model=request.model,
            operation=request.operation,
            timeout_seconds=request.timeout_seconds,
            num_predict=request.num_predict,
            temperature=request.temperature,
            think=False,
            input_value=request.prompt,
            system_prompt=None,
            response_json_schema=request.response_json_schema,
            format_json=request.format_json,
        )
        return GenerationResult(text=text, metrics=metrics)


class OpenAIChatGateway(_OpenAIResponsesAdapter):
    """Chat-shaped text adapter for structured extraction and correction."""

    def generate(self, request: ChatGenerationRequest) -> ChatGenerationResult:
        text, raw_response, metrics = self._generate(
            model=request.model,
            operation=request.operation,
            timeout_seconds=request.timeout_seconds,
            num_predict=request.num_predict,
            temperature=request.temperature,
            think=request.think,
            input_value=request.user_prompt,
            system_prompt=request.system_prompt,
            response_json_schema=request.response_json_schema,
            format_json=request.response_json_schema is not None,
        )
        return ChatGenerationResult(
            text=text,
            metrics=metrics,
            raw_response=raw_response,
        )


class OpenAIMultimodalGateway(_OpenAIResponsesAdapter):
    """Image-aware adapter for the separate multimodal generation contract."""

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": request.prompt}]
        content.extend(
            {
                "type": "input_image",
                "image_url": _image_to_data_url(path),
                "detail": self.image_detail,
            }
            for path in request.image_paths
        )
        text, raw_response, metrics = self._generate(
            model=request.model,
            operation=request.operation,
            timeout_seconds=request.timeout_seconds,
            num_predict=request.num_predict,
            temperature=request.temperature,
            think=request.think,
            input_value=[{"role": "user", "content": content}],
            system_prompt=request.system_prompt,
            response_json_schema=request.response_json_schema,
            format_json=request.format_json,
        )
        return MultimodalGenerationResult(
            text=text,
            metrics=metrics,
            text_source="output_text",
            raw_response=raw_response,
        )


def _response_text(response: Any, raw_response: dict[str, Any]) -> str:
    status = str(getattr(response, "status", None) or raw_response.get("status") or "").lower()
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None) or raw_response.get(
            "incomplete_details"
        )
        raise GenerationIncompleteError(
            f"OpenAI generation was incomplete: {details!r}",
            provider="openai",
        )
    if status and status not in {"completed", "succeeded"}:
        raise GenerationError(
            f"OpenAI generation ended with status {status!r}.",
            provider="openai",
        )

    refusal = _find_refusal(raw_response)
    text = str(
        getattr(response, "output_text", None) or raw_response.get("output_text") or ""
    ).strip()
    if refusal:
        raise GenerationRefusedError(
            f"OpenAI refused the generation request: {refusal}",
            provider="openai",
        )
    if not text:
        raise MalformedGenerationError(
            "OpenAI returned no generated text.",
            provider="openai",
        )
    return text


def _find_refusal(value: Any) -> str | None:
    if isinstance(value, dict):
        if str(value.get("type") or "").lower() == "refusal":
            refusal = str(value.get("refusal") or value.get("content") or "").strip()
            return refusal or "request refused"
        for nested in value.values():
            nested_refusal = _find_refusal(nested)
            if nested_refusal:
                return nested_refusal
    elif isinstance(value, list):
        for nested in value:
            nested_refusal = _find_refusal(nested)
            if nested_refusal:
                return nested_refusal
    return None


def _normalize_openai_error(exc: Exception) -> GenerationError:
    if isinstance(exc, GenerationError):
        return exc
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and 400 <= status_code < 500 and status_code not in {408, 429}:
        return GenerationError(
            f"OpenAI request failed with HTTP {status_code}: {exc}",
            provider="openai",
        )
    return GenerationProviderUnavailableError(
        f"OpenAI request failed: {type(exc).__name__}: {exc}",
        provider="openai",
    )


def _metrics(response: Any, *, model: str, duration_ms: float) -> ModelCallMetrics:
    raw = _response_to_dict(response)
    usage = raw.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return ModelCallMetrics(
        provider="openai",
        endpoint="generate",
        model=str(raw.get("model") or getattr(response, "model", None) or model),
        request_duration_ms=max(0.0, duration_ms),
        prompt_eval_count=_optional_nonnegative_int(usage.get("input_tokens")),
        eval_count=_optional_nonnegative_int(usage.get("output_tokens")),
        done_reason=str(raw.get("status") or getattr(response, "status", None) or "").strip()
        or None,
    )


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return dict(response)
    if hasattr(response, "model_dump"):
        value = response.model_dump()
        return dict(value) if isinstance(value, dict) else {"value": value}
    if hasattr(response, "to_dict"):
        value = response.to_dict()
        return dict(value) if isinstance(value, dict) else {"value": value}
    result: dict[str, Any] = {}
    for name in ("id", "model", "status", "output", "output_text", "usage"):
        value = getattr(response, name, None)
        if value is not None:
            result[name] = value
    return result


def _schema_name(operation: str) -> str:
    value = _SCHEMA_NAME.sub("_", str(operation or "generation")).strip("_")
    return (value or "generation")[:64]


def _reasoning_effort_for_request(
    *,
    model: str,
    configured_effort: str | None,
    think: bool,
) -> str | None:
    """Return a configured effort only when it is valid for a known model family."""

    if configured_effort is None:
        return None

    effort = configured_effort if think else "none"
    model_id = str(model or "").strip().lower()
    if model_id.startswith("gpt-4"):
        return None
    if model_id == "gpt-5-pro" or model_id.startswith("gpt-5-pro-"):
        return effort if effort == "high" else None
    if model_id.startswith("gpt-5."):
        if "-pro" in model_id:
            return effort if effort in {"medium", "high"} else None
        if "-codex" in model_id:
            return effort if effort in _O_SERIES_REASONING_EFFORTS else None
        return effort if effort in _LATER_GPT_5_REASONING_EFFORTS else None
    if model_id == "gpt-5" or model_id.startswith("gpt-5-"):
        if "-codex" in model_id:
            return effort if effort in _O_SERIES_REASONING_EFFORTS else None
        return effort if effort in _ORIGINAL_GPT_5_REASONING_EFFORTS else None
    if _is_standard_o_series_model(model_id):
        return effort if effort in _O_SERIES_REASONING_EFFORTS else None
    return None


def _supports_temperature(*, model: str, reasoning_effort: str | None) -> bool:
    """Return whether the Responses model accepts temperature for this request."""

    model_id = str(model or "").strip().lower()
    if any(
        model_id == family or model_id.startswith(f"{family}-")
        for family in ("o1", "o3", "o4-mini")
    ):
        return False
    if any(
        model_id == family or model_id.startswith(f"{family}-") for family in ("gpt-5.1", "gpt-5.2")
    ):
        return not any(suffix in model_id for suffix in ("-pro", "-codex")) and (
            reasoning_effort in {None, "none"}
        )
    if model_id == "gpt-5" or model_id.startswith(("gpt-5-", "gpt-5.")):
        return False
    return True


def _is_standard_o_series_model(model_id: str) -> bool:
    families = ("o1", "o1-mini", "o3", "o3-mini", "o4-mini")
    return any(
        model_id == family
        or (model_id.startswith(f"{family}-") and model_id.removeprefix(f"{family}-")[:1].isdigit())
        for family in families
    )


def _openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Translate a neutral JSON Schema to OpenAI's strict supported subset."""

    return _normalize_strict_schema_node(schema)


def _has_dynamic_object_shape(value: Any) -> bool:
    """Return whether strict Structured Outputs would close a dynamic object."""

    if isinstance(value, list):
        return any(_has_dynamic_object_shape(item) for item in value)
    if not isinstance(value, dict):
        return False
    if "patternProperties" in value:
        return True
    properties = value.get("properties")
    schema_type = value.get("type")
    is_object = schema_type == "object" or (
        isinstance(schema_type, list) and "object" in schema_type
    )
    if (is_object or isinstance(properties, dict)) and ("additionalProperties" not in value):
        return True
    if "additionalProperties" in value and value["additionalProperties"] is not False:
        return True
    return any(_has_dynamic_object_shape(item) for item in value.values())


def _normalize_strict_schema_node(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_strict_schema_node(item) for item in value]
    if not isinstance(value, dict):
        return value

    original_required = {str(name) for name in value.get("required", []) if isinstance(name, str)}
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if key in _UNSUPPORTED_STRICT_SCHEMA_KEYWORDS or key in {
            "additionalProperties",
            "required",
        }:
            continue
        if key == "properties" and isinstance(item, dict):
            properties: dict[str, Any] = {}
            for name, property_schema in item.items():
                normalized_property = _normalize_strict_schema_node(property_schema)
                if name not in original_required:
                    normalized_property = _make_schema_nullable(normalized_property)
                properties[name] = normalized_property
            normalized[key] = properties
        else:
            normalized[key] = _normalize_strict_schema_node(item)

    properties = normalized.get("properties")
    if normalized.get("type") == "object" or isinstance(properties, dict):
        properties = properties if isinstance(properties, dict) else {}
        normalized["required"] = list(properties)
        normalized["additionalProperties"] = False
    return normalized


def _make_schema_nullable(schema: Any) -> Any:
    if not isinstance(schema, dict) or _schema_allows_null(schema):
        return schema
    if "const" in schema:
        return {"anyOf": [schema, {"type": "null"}]}

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        schema["type"] = [schema_type, "null"]
        enum = schema.get("enum")
        if isinstance(enum, list) and None not in enum:
            enum.append(None)
        return schema
    if isinstance(schema_type, list):
        schema["type"] = [*schema_type, "null"]
        enum = schema.get("enum")
        if isinstance(enum, list) and None not in enum:
            enum.append(None)
        return schema

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        schema["anyOf"] = [*any_of, {"type": "null"}]
        return schema
    return {"anyOf": [schema, {"type": "null"}]}


def _schema_allows_null(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type):
        return True
    enum = schema.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    any_of = schema.get("anyOf")
    return isinstance(any_of, list) and any(
        isinstance(option, dict) and _schema_allows_null(option) for option in any_of
    )


def _image_to_data_url(path: Path) -> str:
    image_path = Path(path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Generation image does not exist: {image_path}")
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        payload = image_path.read_bytes()
        return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"

    from PIL import Image

    with Image.open(image_path) as image:
        converted = image.convert("RGB")
        buffer = io.BytesIO()
        converted.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _optional_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


__all__ = [
    "OpenAIChatGateway",
    "OpenAIGenerationGateway",
    "OpenAIMultimodalGateway",
]
