"""Telemetry decorator for language-model gateways."""

from __future__ import annotations

import time
import uuid

from receipt_intelligence.application.events import ModelCallCompletedEvent
from receipt_intelligence.application.model_call_context import (
    ModelCallContext,
    current_model_call_context,
)
from receipt_intelligence.application.ports.events import EventSink
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    GenerationResult,
    LlmGateway,
    ModelCallMetrics,
)
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
        call_id = f"mc_{uuid.uuid4().hex}"
        started_at = utc_now_iso()
        started = time.perf_counter()
        context = self.default_context.merged(current_model_call_context())
        try:
            result = self.delegate.generate(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            _publish_safely(
                self.event_sink,
                ModelCallCompletedEvent(
                    call_id=call_id,
                    occurred_at=utc_now_iso(),
                    started_at=started_at,
                    trace_id=context.trace_id,
                    job_id=context.job_id,
                    receipt_id=context.receipt_id,
                    query_id=context.query_id,
                    operation=request.operation,
                    provider=_provider_name(self.delegate),
                    model=request.model,
                    endpoint="generate",
                    status="failed",
                    attempt=request.attempt,
                    duration_ms=duration_ms,
                    input_characters=len(request.prompt),
                    output_characters=0,
                    configured_context_window=request.num_ctx,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000.0
        metrics = result.metrics
        _publish_safely(
            self.event_sink,
            _success_event(
                call_id=call_id,
                started_at=started_at,
                occurred_at=utc_now_iso(),
                duration_ms=duration_ms,
                context=context,
                request=request,
                result=result,
                metrics=metrics,
                fallback_provider=_provider_name(self.delegate),
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
    request: GenerationRequest,
    result: GenerationResult,
    metrics: ModelCallMetrics | None,
    fallback_provider: str,
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
        input_characters=len(request.prompt),
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
    )


def _provider_name(delegate: object) -> str:
    name = type(delegate).__name__.lower()
    if "ollama" in name:
        return "ollama"
    return name.removesuffix("gateway") or "unknown"


def _ns_to_ms(value: int | None) -> float | None:
    return None if value is None else round(value / 1_000_000.0, 3)


__all__ = ["ObservedLlmGateway"]
