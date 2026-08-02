from __future__ import annotations

from unittest.mock import patch

from receipt_intelligence.adapters.chat.ollama import OllamaChatGateway
from receipt_intelligence.application.ports.chat import ChatGenerationRequest


def test_ollama_chat_payload_preserves_schema_and_thinking_setting() -> None:
    captured = {}

    def fake_http(url, *, payload, timeout):
        captured.update(url=url, payload=payload, timeout=timeout)
        return {"model": "gemma4", "message": {"content": '{"value":1}'}, "eval_count": 4}

    request = ChatGenerationRequest(
        model="gemma4",
        system_prompt="system",
        user_prompt="user",
        response_json_schema={"type": "object"},
        think=False,
        seed=42,
    )
    with patch("receipt_intelligence.adapters.chat.ollama._http_json", fake_http):
        result = OllamaChatGateway("http://ollama").generate(request)
    assert result.text == '{"value":1}'
    assert captured["payload"]["think"] is False
    assert captured["payload"]["format"] == {"type": "object"}
    assert captured["payload"]["options"]["seed"] == 42
