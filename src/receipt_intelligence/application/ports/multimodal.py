"""Provider-neutral multimodal generation contracts.

This port is intentionally separate from :mod:`receipt_intelligence.application.ports.llm`.
Text-only and image-aware model calls have different transport and lifecycle concerns, and
keeping them separate prevents optional image fields from leaking into every LLM request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from receipt_intelligence.application.ports.llm import ModelCallMetrics


@dataclass(frozen=True, slots=True)
class MultimodalGenerationRequest:
    """One image-aware generation request.

    ``response_json_schema`` is optional because transcription normally returns text. Later
    extraction stages may still use the same port for image-grounded structured responses.
    """

    model: str
    prompt: str
    image_paths: tuple[Path, ...]
    operation: str = "multimodal_generation"
    attempt: int = 1
    think: bool = False
    num_ctx: int = 8192
    num_predict: int = 4096
    temperature: float = 0.0
    keep_alive: str | None = None
    timeout_seconds: float = 300.0
    format_json: bool = False
    response_json_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        model = str(self.model or "").strip()
        prompt = str(self.prompt or "").strip()
        operation = str(self.operation or "").strip()
        image_paths = tuple(Path(path) for path in self.image_paths)

        if not model:
            raise ValueError("MultimodalGenerationRequest.model must not be empty.")
        if not prompt:
            raise ValueError("MultimodalGenerationRequest.prompt must not be empty.")
        if not operation:
            raise ValueError("MultimodalGenerationRequest.operation must not be empty.")
        if not image_paths:
            raise ValueError("MultimodalGenerationRequest.image_paths must not be empty.")
        if self.attempt < 1:
            raise ValueError("MultimodalGenerationRequest.attempt must be >= 1.")
        if self.num_ctx < 1 or self.num_predict < 1:
            raise ValueError("Multimodal token limits must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("Multimodal timeout_seconds must be positive.")

        schema = self.response_json_schema
        if schema is not None:
            if not isinstance(schema, dict) or not schema:
                raise ValueError("response_json_schema must be a non-empty object.")
            if not self.format_json:
                raise ValueError("response_json_schema requires format_json=True.")
            object.__setattr__(self, "response_json_schema", dict(schema))

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "image_paths", image_paths)


@dataclass(frozen=True, slots=True)
class MultimodalGenerationResult:
    """Provider-neutral result of one multimodal model call."""

    text: str
    metrics: ModelCallMetrics | None = None

    def __post_init__(self) -> None:
        text = str(self.text or "").strip()
        if not text:
            raise ValueError("MultimodalGenerationResult.text must not be empty.")
        object.__setattr__(self, "text", text)


class MultimodalGateway(Protocol):
    """Port implemented by Ollama or another multimodal model adapter."""

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult: ...


__all__ = [
    "MultimodalGateway",
    "MultimodalGenerationRequest",
    "MultimodalGenerationResult",
]
