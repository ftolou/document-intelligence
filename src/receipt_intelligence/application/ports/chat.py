"""Provider-neutral structured chat-generation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from receipt_intelligence.application.ports.llm import ModelCallMetrics


@dataclass(frozen=True, slots=True)
class ChatGenerationRequest:
    model: str
    system_prompt: str
    user_prompt: str
    response_json_schema: dict[str, Any]
    operation: str = "chat_generation"
    attempt: int = 1
    think: bool = False
    num_ctx: int = 16384
    num_predict: int = 4096
    temperature: float = 0.0
    seed: int = 42
    keep_alive: str | None = None
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        model = str(self.model or "").strip()
        system_prompt = str(self.system_prompt or "").strip()
        user_prompt = str(self.user_prompt or "").strip()
        operation = str(self.operation or "").strip()
        if not model or not system_prompt or not user_prompt or not operation:
            raise ValueError("ChatGenerationRequest text fields must not be empty.")
        if self.attempt < 1:
            raise ValueError("ChatGenerationRequest.attempt must be positive.")
        if self.num_ctx < 1 or self.num_predict < 1:
            raise ValueError("Chat token limits must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("Chat timeout_seconds must be positive.")
        if not isinstance(self.response_json_schema, dict) or not self.response_json_schema:
            raise ValueError("response_json_schema must be a non-empty object.")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "system_prompt", system_prompt)
        object.__setattr__(self, "user_prompt", user_prompt)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "response_json_schema", dict(self.response_json_schema))


@dataclass(frozen=True, slots=True)
class ChatGenerationResult:
    text: str
    thinking: str | None = None
    metrics: ModelCallMetrics | None = None
    raw_response: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        text = str(self.text or "").strip()
        if not text:
            raise ValueError("ChatGenerationResult.text must not be empty.")
        object.__setattr__(self, "text", text)
        if self.thinking is not None:
            object.__setattr__(self, "thinking", str(self.thinking).strip() or None)
        if self.raw_response is not None:
            object.__setattr__(self, "raw_response", dict(self.raw_response))


class ChatGateway(Protocol):
    def generate(self, request: ChatGenerationRequest) -> ChatGenerationResult: ...


__all__ = ["ChatGateway", "ChatGenerationRequest", "ChatGenerationResult"]
