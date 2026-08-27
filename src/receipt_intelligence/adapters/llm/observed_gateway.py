"""Telemetry decorator for language-model gateways."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from typing import Protocol, TypeVar

from receipt_intelligence.application.events import ModelCallCompletedEvent
from receipt_intelligence.application.model_call_context import (
    ModelCallContext,
    current_model_call_context,
)
from receipt_intelligence.application.ports.chat import (
    ChatGateway,
    ChatGenerationRequest,
    ChatGenerationResult,
)
from receipt_intelligence.application.ports.events import EventSink
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    GenerationResult,
    LlmGateway,
    ModelCallMetrics,
)
from receipt_intelligence.application.ports.multimodal import (
    MultimodalGateway,
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)
from receipt_intelligence.application.query_diagnostics import record_query_diagnostic
from receipt_intelligence.observability.timing import utc_now_iso


class ObservedLlmGateway:
    """Publish one neutral event around every delegated generation request."""

    def __init__(
        self,
        delegate: LlmGateway,
        event_sink: EventSink,
        *,
        default_context: ModelCallContext | None = None,
    ) -> None:
        self.delegate = delegate
        self.event_sink = event_sink
        self.default_context = default_context or ModelCallContext()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return _observed_generate(
            delegate=self.delegate,
            event_sink=self.event_sink,
            default_context=self.default_context,
            request=request,
            input_characters=len(request.prompt),
            request_diagnostics={
                "format_json": request.format_json,
                "response_json_schema": request.response_json_schema,
                "prompt": request.prompt,
            },
            attributes=None,
            invoke=lambda: self.delegate.generate(request),
        )


class ObservedChatGateway:
    """Publish one neutral event around every structured chat request."""

    def __init__(
        self,
        delegate: ChatGateway,
        event_sink: EventSink,
        *,
        default_context: ModelCallContext | None = None,
    ) -> None:
        self.delegate = delegate
        self.event_sink = event_sink
        self.default_context = default_context or ModelCallContext()

    def generate(self, request: ChatGenerationRequest) -> ChatGenerationResult:
        system_characters = len(request.system_prompt or "")
        return _observed_generate(
            delegate=self.delegate,
            event_sink=self.event_sink,
            default_context=self.default_context,
            request=request,
            input_characters=system_characters + len(request.user_prompt),
            request_diagnostics={
                "system_prompt": request.system_prompt,
                "prompt": request.user_prompt,
                "think": request.think,
                "response_json_schema": request.response_json_schema,
            },
            attributes={
                "modality": "chat",
                "think": request.think,
                "system_prompt_characters": system_characters,
            },
            invoke=lambda: self.delegate.generate(request),
        )


class ObservedMultimodalGateway:
    """Publish one neutral event around every image-aware model request."""

    def __init__(
        self,
        delegate: MultimodalGateway,
        event_sink: EventSink,
        *,
        default_context: ModelCallContext | None = None,
    ) -> None:
        self.delegate = delegate
        self.event_sink = event_sink
        self.default_context = default_context or ModelCallContext()

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        return _observed_generate(
            delegate=self.delegate,
            event_sink=self.event_sink,
            default_context=self.default_context,
            request=request,
            input_characters=len(request.prompt),
            request_diagnostics={
                "prompt": request.prompt,
                "think": request.think,
                "format_json": request.format_json,
                "response_json_schema": request.response_json_schema,
                "image_paths": [str(path) for path in request.image_paths],
            },
            attributes={
                "modality": "multimodal",
                "think": request.think,
                "image_count": len(request.image_paths),
            },
            invoke=lambda: self.delegate.generate(request),
        )


class _ObservedRequest(Protocol):
    model: str
    operation: str
    attempt: int
    num_ctx: int
    num_predict: int
    temperature: float | None
    timeout_seconds: float


class _ObservedResult(Protocol):
    text: str
    metrics: ModelCallMetrics | None


_ResultT = TypeVar("_ResultT", bound=_ObservedResult)


def _observed_generate(
    *,
    delegate: object,
    event_sink: EventSink,
    default_context: ModelCallContext,
    request: _ObservedRequest,
    input_characters: int,
    request_diagnostics: Mapping[str, object],
    attributes: Mapping[str, object] | None,
    invoke: Callable[[], _ResultT],
) -> _ResultT:
    call_id = f"mc_{uuid.uuid4().hex}"
    started_at = utc_now_iso()
    started = time.perf_counter()
    context = default_context.merged(current_model_call_context())
    record_query_diagnostic(
        "llm.request",
        {
            "call_id": call_id,
            "trace_id": context.trace_id,
            "query_id": context.query_id,
            "operation": request.operation,
            "attempt": request.attempt,
            "model": request.model,
            "num_ctx": request.num_ctx,
            "num_predict": request.num_predict,
            "temperature": request.temperature,
            "timeout_seconds": request.timeout_seconds,
            **request_diagnostics,
        },
    )
    try:
        result = invoke()
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000.0
        error = f"{type(exc).__name__}: {exc}"
        record_query_diagnostic(
            "llm.response",
            {
                "call_id": call_id,
                "operation": request.operation,
                "attempt": request.attempt,
                "status": "failed",
                "duration_ms": round(duration_ms, 3),
                "error": error,
            },
        )
        _publish_safely(
            event_sink,
            ModelCallCompletedEvent(
                call_id=call_id,
                occurred_at=utc_now_iso(),
                started_at=started_at,
                trace_id=context.trace_id,
                job_id=context.job_id,
                receipt_id=context.receipt_id,
                query_id=context.query_id,
                operation=request.operation,
                provider=_provider_name(delegate),
                model=request.model,
                endpoint="generate",
                status="failed",
                attempt=request.attempt,
                duration_ms=duration_ms,
                input_characters=input_characters,
                output_characters=0,
                configured_context_window=request.num_ctx,
                error=error,
                attributes=attributes,
            ),
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1000.0
    metrics = result.metrics
    record_query_diagnostic(
        "llm.response",
        {
            "call_id": call_id,
            "operation": request.operation,
            "attempt": request.attempt,
            "status": "completed",
            "duration_ms": round(duration_ms, 3),
            "response_text": result.text,
            "metrics": metrics.to_diagnostics() if metrics is not None else None,
        },
    )
    _publish_safely(
        event_sink,
        _success_event(
            call_id=call_id,
            started_at=started_at,
            occurred_at=utc_now_iso(),
            duration_ms=duration_ms,
            context=context,
            request=request,
            result=result,
            metrics=metrics,
            fallback_provider=_provider_name(delegate),
            input_characters=input_characters,
            attributes=attributes,
        ),
    )
    return result


def _publish_safely(event_sink: EventSink, event: ModelCallCompletedEvent) -> None:
    try:
        event_sink.publish(event)
    except Exception:
        # Telemetry is observational and must not alter model-call behavior.
        return


def _success_event(
    *,
    call_id: str,
    started_at: str,
    occurred_at: str,
    duration_ms: float,
    context: ModelCallContext,
    request: _ObservedRequest,
    result: _ObservedResult,
    metrics: ModelCallMetrics | None,
    fallback_provider: str,
    input_characters: int,
    attributes: Mapping[str, object] | None,
) -> ModelCallCompletedEvent:
    return ModelCallCompletedEvent(
        call_id=call_id,
        occurred_at=occurred_at,
        started_at=started_at,
        trace_id=context.trace_id,
        job_id=context.job_id,
        receipt_id=context.receipt_id,
        query_id=context.query_id,
        operation=request.operation,
        provider=metrics.provider if metrics is not None else fallback_provider,
        model=metrics.model if metrics is not None else request.model,
        endpoint=metrics.endpoint if metrics is not None else "generate",
        status="completed",
        attempt=request.attempt,
        duration_ms=metrics.request_duration_ms if metrics is not None else duration_ms,
        input_tokens=metrics.prompt_eval_count if metrics is not None else None,
        output_tokens=metrics.eval_count if metrics is not None else None,
        input_characters=input_characters,
        output_characters=len(result.text),
        token_source="provider_reported" if metrics is not None else "unavailable",
        model_total_duration_ms=_ns_to_ms(
            metrics.total_duration_ns if metrics is not None else None
        ),
        model_load_duration_ms=_ns_to_ms(metrics.load_duration_ns if metrics is not None else None),
        prompt_evaluation_duration_ms=_ns_to_ms(
            metrics.prompt_eval_duration_ns if metrics is not None else None
        ),
        generation_duration_ms=_ns_to_ms(metrics.eval_duration_ns if metrics is not None else None),
        configured_context_window=request.num_ctx,
        stop_reason=metrics.done_reason if metrics is not None else None,
        attributes=attributes,
    )


def _provider_name(delegate: object) -> str:
    name = type(delegate).__name__.lower()
    if "ollama" in name:
        return "ollama"
    return name.removesuffix("gateway") or "unknown"


def _ns_to_ms(value: int | None) -> float | None:
    return None if value is None else round(value / 1_000_000.0, 3)


__all__ = ["ObservedChatGateway", "ObservedLlmGateway", "ObservedMultimodalGateway"]
