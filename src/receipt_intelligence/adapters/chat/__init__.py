"""Structured chat-model adapters."""

from receipt_intelligence.adapters.chat.ollama import OllamaChatGateway
from receipt_intelligence.adapters.llm.openai_responses import OpenAIChatGateway

__all__ = ["OllamaChatGateway", "OpenAIChatGateway"]
