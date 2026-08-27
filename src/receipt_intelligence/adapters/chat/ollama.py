"""Ollama `/api/chat` adapter for structured and unformatted Gemma calls."""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from receipt_intelligence.adapters.llm.ollama_gateway import (
    model_metrics_from_ollama_payload,
    normalize_ollama_error,
    validate_ollama_completion,
)
from receipt_intelligence.application.ports.chat import (
    ChatGateway,
    ChatGenerationRequest,
    ChatGenerationResult,
)
from receipt_intelligence.application.ports.llm import MalformedGenerationError


class OllamaChatGateway(ChatGateway):
    def __init__(self, base_url: str) -> None:
        normalized = str(base_url or "").strip().rstrip("/")
        if not normalized:
            raise ValueError("base_url must not be empty.")
        self.base_url = normalized

    def generate(self, request: ChatGenerationRequest) -> ChatGenerationResult:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})
        options: dict[str, Any] = {
            "seed": request.seed,
            "num_ctx": request.num_ctx,
            "num_predict": request.num_predict,
        }
        if request.temperature is not None:
            options["temperature"] = request.temperature
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "think": request.think,
            "options": options,
        }
        if request.response_json_schema is not None:
            payload["format"] = request.response_json_schema
        if request.keep_alive not in (None, ""):
            payload["keep_alive"] = request.keep_alive

        started = time.perf_counter()
        try:
            response = _http_json(
                f"{self.base_url}/api/chat",
                payload=payload,
                timeout=request.timeout_seconds,
            )
        except Exception as exc:
            raise normalize_ollama_error(exc) from exc
        duration_ms = (time.perf_counter() - started) * 1000.0
        validate_ollama_completion(response)
        message = response.get("message")
        if not isinstance(message, dict):
            raise MalformedGenerationError(
                "Ollama chat response has no message object.",
                provider="ollama",
            )
        text = str(message.get("content") or "").strip()
        if not text:
            raise MalformedGenerationError(
                "Ollama chat returned empty content.",
                provider="ollama",
            )
        return ChatGenerationResult(
            text=text,
            thinking=(str(message.get("thinking") or "").strip() or None),
            metrics=model_metrics_from_ollama_payload(
                response,
                endpoint="generate",
                model=request.model,
                request_duration_ms=duration_ms,
            ),
            raw_response=response,
        )


def _http_json(url: str, *, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object from {url}.")
    return value


__all__ = ["OllamaChatGateway"]
