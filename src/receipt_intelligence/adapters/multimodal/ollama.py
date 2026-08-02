"""Ollama chat adapter for image-aware generation."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from receipt_intelligence.adapters.llm.ollama_gateway import model_metrics_from_ollama_payload
from receipt_intelligence.application.ports.multimodal import (
    MultimodalGateway,
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)


class OllamaMultimodalGateway(MultimodalGateway):
    def __init__(self, base_url: str) -> None:
        normalized = str(base_url or "").strip().rstrip("/")
        if not normalized:
            raise ValueError("base_url must not be empty.")
        self.base_url = normalized

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        images = [_encode_image(path) for path in request.image_paths]
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_ctx": request.num_ctx,
            "num_predict": request.num_predict,
        }
        options.update(request.provider_options)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt,
                    "images": images,
                }
            ],
            "stream": False,
            "think": request.think,
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
                f"{self.base_url}/api/chat",
                payload=payload,
                timeout=request.timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {message}") from exc
        duration_ms = (time.perf_counter() - started) * 1000.0
        text, source = _extract_text(response)
        return MultimodalGenerationResult(
            text=text,
            text_source=source,
            raw_response=response,
            metrics=model_metrics_from_ollama_payload(
                response,
                endpoint="generate",
                model=request.model,
                request_duration_ms=duration_ms,
            ),
        )


def _extract_text(response: dict[str, Any]) -> tuple[str, str]:
    message = response.get("message")
    if isinstance(message, dict):
        for field_name in ("content", "thinking"):
            value = message.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip(), f"message.{field_name}"
    legacy = response.get("response")
    if isinstance(legacy, str) and legacy.strip():
        return legacy.strip(), "response"
    raise RuntimeError(
        "Ollama multimodal response contained no nonempty text in "
        "message.content, message.thinking, or response."
    )


def _encode_image(path: Path) -> str:
    image_path = Path(path)
    try:
        data = image_path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Multimodal image not found: {image_path}") from exc
    return base64.b64encode(data).decode("ascii")


def _http_json(
    url: str,
    *,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return obj


__all__ = ["OllamaMultimodalGateway"]
