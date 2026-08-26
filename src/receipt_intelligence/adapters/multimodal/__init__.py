"""Multimodal model adapters."""

from receipt_intelligence.adapters.llm.openai_responses import OpenAIMultimodalGateway
from receipt_intelligence.adapters.multimodal.ollama import OllamaMultimodalGateway

__all__ = ["OllamaMultimodalGateway", "OpenAIMultimodalGateway"]
