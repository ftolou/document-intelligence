"""Language-model adapters."""

from receipt_intelligence.adapters.llm.observed_gateway import (
    ObservedChatGateway,
    ObservedLlmGateway,
    ObservedMultimodalGateway,
)
from receipt_intelligence.adapters.llm.ollama_gateway import (
    OllamaGateway,
    model_metrics_from_ollama_payload,
)
from receipt_intelligence.adapters.llm.openai_responses import (
    OpenAIChatGateway,
    OpenAIGenerationGateway,
    OpenAIMultimodalGateway,
)

__all__ = [
    "ObservedChatGateway",
    "ObservedLlmGateway",
    "ObservedMultimodalGateway",
    "OllamaGateway",
    "OpenAIChatGateway",
    "OpenAIGenerationGateway",
    "OpenAIMultimodalGateway",
    "model_metrics_from_ollama_payload",
]
