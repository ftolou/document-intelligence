"""Ollama implementation of the provider-neutral LLM gateway."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Literal

from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    GenerationResult,
    LlmGateway,
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
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_ctx": request.num_ctx,
                "num_predict": request.num_predict,
            },
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
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {message}") from exc
        duration_ms = (time.perf_counter() - started) * 1000.0
        text = str(response.get("response") or "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response")
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
        prompt_eval_duration_ns=_optional_nonnegative_int(
            payload.get("prompt_eval_duration")
        ),
        eval_count=(
            _optional_nonnegative_int(payload.get("eval_count"))
            if endpoint == "generate"
            else None
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


__all__ = ["OllamaGateway", "model_metrics_from_ollama_payload"]
