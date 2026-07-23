from __future__ import annotations

from receipt_intelligence.observability.ollama import (
    OllamaCallMetrics,
    OllamaTextResponse,
    get_ollama_metrics,
)


def test_generate_metrics_expose_load_prompt_and_generation_durations() -> None:
    metrics = OllamaCallMetrics.from_payload(
        {
            "model": "gemma4",
            "total_duration": 12_000_000_000,
            "load_duration": 2_000_000_000,
            "prompt_eval_count": 300,
            "prompt_eval_duration": 3_000_000_000,
            "eval_count": 140,
            "eval_duration": 7_000_000_000,
            "done_reason": "stop",
        },
        endpoint="generate",
        model="gemma4",
        request_duration_ms=12_500.0,
    )

    diagnostics = metrics.to_diagnostics()

    assert diagnostics["load_duration_ms"] == 2000.0
    assert diagnostics["prompt_eval_duration_ms"] == 3000.0
    assert diagnostics["eval_duration_ms"] == 7000.0
    assert diagnostics["prompt_tokens_per_second"] == 100.0
    assert diagnostics["generated_tokens_per_second"] == 20.0


def test_text_response_remains_string_compatible() -> None:
    metrics = OllamaCallMetrics(
        endpoint="generate",
        model="gemma4",
        request_duration_ms=1.0,
    )
    response = OllamaTextResponse('{"ok": true}', metrics=metrics)

    assert isinstance(response, str)
    assert response.startswith("{")
    assert get_ollama_metrics(response) == metrics
    assert get_ollama_metrics('{"ok": true}') is None
