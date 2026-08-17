from __future__ import annotations

import copy

from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports.llm import GenerationResult, ModelCallMetrics


def test_generation_result_deepcopy_is_a_normal_value_object() -> None:
    metrics = ModelCallMetrics(
        provider="ollama",
        endpoint="generate",
        model="gemma4",
        request_duration_ms=12.5,
    )
    original = GenerationResult(text='{"ok":true}', metrics=metrics)

    copied = copy.deepcopy(original)

    assert copied == original
    assert copied.metrics == metrics


def test_json_parser_accepts_explicit_generation_result() -> None:
    result = GenerationResult(text='```json\n{"status":"ok"}\n```')

    assert parse_json_from_llm(result) == {"status": "ok"}
