from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from receipt_intelligence.adapters.llm import (
    ObservedChatGateway,
    ObservedLlmGateway,
    ObservedMultimodalGateway,
)
from receipt_intelligence.application.model_call_context import bind_model_call_context
from receipt_intelligence.application.ports.chat import (
    ChatGenerationRequest,
    ChatGenerationResult,
)
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    GenerationResult,
    ModelCallMetrics,
)
from receipt_intelligence.application.ports.multimodal import (
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)
from receipt_intelligence.application.query_diagnostics import capture_query_diagnostics


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


class SuccessfulChatGateway:
    def generate(self, request: ChatGenerationRequest) -> ChatGenerationResult:
        return ChatGenerationResult(
            text='{"items": []}',
            metrics=ModelCallMetrics(
                provider="ollama",
                endpoint="generate",
                model=request.model,
                request_duration_ms=75.0,
                prompt_eval_count=500,
                eval_count=40,
            ),
        )


class SuccessfulMultimodalGateway:
    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        return MultimodalGenerationResult(
            text="TOTAL 12,34",
            metrics=ModelCallMetrics(
                provider="ollama",
                endpoint="generate",
                model=request.model,
                request_duration_ms=90.0,
                prompt_eval_count=600,
                eval_count=30,
            ),
        )


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


def test_observed_gateway_captures_prompt_and_raw_response_when_query_logging_is_enabled() -> None:
    sink = RecordingSink()
    gateway = ObservedLlmGateway(SuccessfulGateway(), sink)

    with capture_query_diagnostics(enabled=True) as diagnostics:
        gateway.generate(
            GenerationRequest(
                model="gemma4",
                prompt="classify candidate c001",
                operation="rag_candidate_resolution",
            )
        )

    records = diagnostics.snapshot()
    assert records[0]["event"] == "llm.request"
    assert records[0]["prompt"] == "classify candidate c001"
    assert records[1]["event"] == "llm.response"
    assert records[1]["response_text"] == '{"ok": true}'
    assert records[1]["metrics"]["eval_count"] == 120


def test_observed_chat_gateway_publishes_thinking_and_context() -> None:
    sink = RecordingSink()
    gateway = ObservedChatGateway(SuccessfulChatGateway(), sink)

    with bind_model_call_context(trace_id="receipt-trace", job_id="receipt-job"):
        gateway.generate(
            ChatGenerationRequest(
                model="gemma4",
                system_prompt="system",
                user_prompt="receipt rows",
                operation="receipt_structured_items",
                think=True,
                num_ctx=16384,
            )
        )

    record = sink.records[0]
    assert record["trace_id"] == "receipt-trace"
    assert record["job_id"] == "receipt-job"
    assert record["operation"] == "receipt_structured_items"
    assert record["input_tokens"] == 500
    assert record["output_tokens"] == 40
    assert record["input_characters"] == len("systemreceipt rows")
    assert record["configured_context_window"] == 16384
    assert record["attributes"] == {
        "modality": "chat",
        "think": True,
        "system_prompt_characters": len("system"),
    }


def test_observed_multimodal_gateway_publishes_image_metadata() -> None:
    sink = RecordingSink()
    gateway = ObservedMultimodalGateway(SuccessfulMultimodalGateway(), sink)

    gateway.generate(
        MultimodalGenerationRequest(
            model="qwen3-vl",
            prompt="transcribe",
            image_paths=(Path("receipt.png"),),
            operation="receipt_transcription",
        )
    )

    record = sink.records[0]
    assert record["operation"] == "receipt_transcription"
    assert record["input_tokens"] == 600
    assert record["output_tokens"] == 30
    assert record["attributes"] == {
        "modality": "multimodal",
        "think": False,
        "image_count": 1,
    }
