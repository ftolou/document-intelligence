from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from receipt_intelligence.adapters.llm import ObservedLlmGateway
from receipt_intelligence.application.model_call_context import bind_model_call_context
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    GenerationResult,
    ModelCallMetrics,
)


@dataclass
class RecordingSink:
    records: list[dict[str, object]] = field(default_factory=list)

    def publish(self, event: object) -> None:
        self.records.append(event.to_record())  # type: ignore[attr-defined]


class SuccessfulGateway:
    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            text='{"ok": true}',
            metrics=ModelCallMetrics(
                provider="ollama",
                endpoint="generate",
                model=request.model,
                request_duration_ms=125.0,
                prompt_eval_count=800,
                prompt_eval_duration_ns=2_000_000_000,
                eval_count=120,
                eval_duration_ns=1_000_000_000,
            ),
        )


class FailingGateway:
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise RuntimeError("provider unavailable")


def test_observed_gateway_publishes_tokens_timing_and_context() -> None:
    sink = RecordingSink()
    gateway = ObservedLlmGateway(SuccessfulGateway(), sink)

    with bind_model_call_context(trace_id="trace-1", job_id="job-1"):
        result = gateway.generate(
            GenerationRequest(
                model="gemma4",
                prompt="receipt evidence",
                operation="receipt_main_parse",
                attempt=2,
            )
        )

    assert result.text == '{"ok": true}'
    record = sink.records[0]
    assert record["trace_id"] == "trace-1"
    assert record["job_id"] == "job-1"
    assert record["operation"] == "receipt_main_parse"
    assert record["attempt"] == 2
    assert record["input_tokens"] == 800
    assert record["output_tokens"] == 120
    assert record["prompt_evaluation_duration_ms"] == 2000.0
    assert record["generation_duration_ms"] == 1000.0
    assert record["status"] == "completed"


def test_observed_gateway_records_failed_calls_without_hiding_exception() -> None:
    sink = RecordingSink()
    gateway = ObservedLlmGateway(FailingGateway(), sink)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        gateway.generate(
            GenerationRequest(
                model="gemma4",
                prompt="receipt evidence",
                operation="receipt_main_parse",
            )
        )

    assert sink.records[0]["status"] == "failed"
    assert "provider unavailable" in str(sink.records[0]["error"])
