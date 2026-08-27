"""Ollama implementation of the provider-neutral LLM gateway."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Literal

from receipt_intelligence.application.ports.llm import (
    GenerationError,
    GenerationIncompleteError,
    GenerationProviderUnavailableError,
    GenerationRequest,
    GenerationResult,
    LlmGateway,
    MalformedGenerationError,
    ModelCallMetrics,
)


class OllamaGateway(LlmGateway):
    def __init__(self, base_url: str) -> None:
        normalized = str(base_url or "").strip().rstrip("/")
        if not normalized:
            raise ValueError("base_url must not be empty.")
        self.base_url = normalized

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            _http_json(
                f"{self.base_url}/api/tags",
                timeout=min(request.timeout_seconds, 20.0),
            )
        except Exception as exc:
            raise GenerationProviderUnavailableError(
                f"Ollama is not reachable at {self.base_url}: {type(exc).__name__}: {exc}",
                provider="ollama",
            ) from exc

        options: dict[str, Any] = {
            "num_ctx": request.num_ctx,
            "num_predict": request.num_predict,
        }
        if request.temperature is not None:
            options["temperature"] = request.temperature
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": options,
        }
        if request.keep_alive not in (None, ""):
            payload["keep_alive"] = request.keep_alive
        if request.response_json_schema is not None:
            payload["format"] = request.response_json_schema
        elif request.format_json:
            payload["format"] = "json"

        started = time.perf_counter()
        try:
            response = _http_json(
                f"{self.base_url}/api/generate",
                payload=payload,
                timeout=request.timeout_seconds,
            )
        except Exception as exc:
            raise normalize_ollama_error(exc) from exc
        duration_ms = (time.perf_counter() - started) * 1000.0
        validate_ollama_completion(response)
        text = str(response.get("response") or "").strip()
        if not text:
            raise MalformedGenerationError(
                "Ollama returned an empty response.",
                provider="ollama",
            )
        return GenerationResult(
            text=text,
            metrics=model_metrics_from_ollama_payload(
                response,
                endpoint="generate",
                model=request.model,
                request_duration_ms=duration_ms,
            ),
        )


def model_metrics_from_ollama_payload(
    payload: dict[str, Any],
    *,
    endpoint: Literal["generate", "embed"],
    model: str,
    request_duration_ms: float,
    input_count: int | None = None,
) -> ModelCallMetrics:
    if endpoint not in {"generate", "embed"}:
        raise ValueError(f"Unsupported Ollama endpoint: {endpoint!r}")
    return ModelCallMetrics(
        provider="ollama",
        endpoint=endpoint,
        model=str(payload.get("model") or model),
        request_duration_ms=max(0.0, float(request_duration_ms)),
        total_duration_ns=_optional_nonnegative_int(payload.get("total_duration")),
        load_duration_ns=_optional_nonnegative_int(payload.get("load_duration")),
        prompt_eval_count=_optional_nonnegative_int(payload.get("prompt_eval_count")),
        prompt_eval_duration_ns=_optional_nonnegative_int(payload.get("prompt_eval_duration")),
        eval_count=(
            _optional_nonnegative_int(payload.get("eval_count")) if endpoint == "generate" else None
        ),
        eval_duration_ns=(
            _optional_nonnegative_int(payload.get("eval_duration"))
            if endpoint == "generate"
            else None
        ),
        done_reason=(
            str(payload.get("done_reason") or "").strip() or None
            if endpoint == "generate"
            else None
        ),
        input_count=input_count,
    )


def normalize_ollama_error(exc: Exception) -> GenerationError:
    """Translate Ollama transport failures into stable Core failures."""

    if isinstance(exc, GenerationError):
        return exc
    if isinstance(exc, urllib.error.HTTPError):
        message = exc.read().decode("utf-8", errors="replace")[:1000]
        error_type = GenerationProviderUnavailableError if exc.code >= 500 else GenerationError
        return error_type(
            f"Ollama HTTP error {exc.code}: {message}",
            provider="ollama",
        )
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return GenerationProviderUnavailableError(
            f"Ollama request failed: {type(exc).__name__}: {exc}",
            provider="ollama",
        )
    if isinstance(exc, ValueError):
        return MalformedGenerationError(
            f"Ollama returned a malformed response: {exc}",
            provider="ollama",
        )
    return GenerationError(
        f"Ollama request failed: {type(exc).__name__}: {exc}",
        provider="ollama",
    )


def validate_ollama_completion(response: dict[str, Any]) -> None:
    if response.get("done") is False:
        raise GenerationIncompleteError(
            f"Ollama generation was incomplete: {response.get('done_reason')!r}",
            provider="ollama",
        )


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return obj


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


__all__ = [
    "OllamaGateway",
    "model_metrics_from_ollama_payload",
    "normalize_ollama_error",
    "validate_ollama_completion",
]
