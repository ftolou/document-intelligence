from __future__ import annotations

from receipt_intelligence.extraction.parsing import llm_parser
from receipt_intelligence.adapters.llm import ollama_gateway


def test_ollama_generate_attaches_provider_metrics(monkeypatch) -> None:
    responses = iter(
        [
            {"models": []},
            {
                "model": "gemma4",
                "response": '{"status":"ok"}',
                "total_duration": 5_000_000_000,
                "load_duration": 1_500_000_000,
                "prompt_eval_count": 120,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 40,
                "eval_duration": 2_000_000_000,
                "done_reason": "stop",
            },
        ]
    )
    monkeypatch.setattr(ollama_gateway, "_http_json", lambda *args, **kwargs: next(responses))

    result = llm_parser.ollama_generate(
        ollama_url="http://localhost:11434",
        model="gemma4",
        prompt="test",
        num_ctx=128,
        num_predict=32,
    )

    metrics = result.metrics
    assert result.text == '{"status":"ok"}'
    assert metrics is not None
    assert metrics.model == "gemma4"
    assert metrics.load_duration_ns == 1_500_000_000
    assert metrics.prompt_eval_count == 120
    assert metrics.eval_count == 40


def test_ollama_generate_sends_residency_options(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object] | None]] = []

    def fake_http_json(
        url: str,
        payload: dict[str, object] | None = None,
        timeout: float = 180.0,
    ) -> dict[str, object]:
        del timeout
        calls.append((url, payload))
        if url.endswith("/api/tags"):
            return {"models": []}
        return {
            "model": "gemma4",
            "response": '{"status":"ok"}',
            "done_reason": "stop",
        }

    monkeypatch.setattr(ollama_gateway, "_http_json", fake_http_json)

    llm_parser.ollama_generate(
        ollama_url="http://localhost:11434",
        model="gemma4",
        prompt="test",
        num_ctx=6144,
        num_predict=128,
        keep_alive="30m",
    )

    generate_payload = calls[1][1]
    assert generate_payload is not None
    assert generate_payload["keep_alive"] == "30m"
    assert generate_payload["options"] == {
        "temperature": 0.0,
        "num_ctx": 6144,
        "num_predict": 128,
    }


def test_ollama_generate_sends_json_schema_as_format_object(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object] | None]] = []

    def fake_http_json(
        url: str,
        payload: dict[str, object] | None = None,
        timeout: float = 180.0,
    ) -> dict[str, object]:
        del timeout
        calls.append((url, payload))
        if url.endswith("/api/tags"):
            return {"models": []}
        return {
            "model": "gemma4",
            "response": '{"schema_version":"v14_6_llm_receipt_1"}',
            "done_reason": "stop",
        }

    monkeypatch.setattr(ollama_gateway, "_http_json", fake_http_json)
    schema = {
        "type": "object",
        "properties": {"schema_version": {"type": "string"}},
        "required": ["schema_version"],
    }

    llm_parser.ollama_generate(
        ollama_url="http://localhost:11434",
        model="gemma4",
        prompt="test",
        response_json_schema=schema,
    )

    generate_payload = calls[1][1]
    assert generate_payload is not None
    assert generate_payload["format"] == schema
