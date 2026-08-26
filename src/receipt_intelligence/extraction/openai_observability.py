"""Composition and observability for the OpenAI one-shot receipt adapter.

The extraction workflow uses the provider-neutral multimodal port. This module
owns provider client construction and reconnects the concrete adapter call to
the application's existing model-call telemetry and extraction-run metrics.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from receipt_intelligence.adapters.observability import JsonFileEventSink
from receipt_intelligence.adapters.storage.sqlite.model_calls import SQLiteModelCallRepository
from receipt_intelligence.application.events import ExtractionRunEvent, ModelCallCompletedEvent
from receipt_intelligence.application.ports.events import EventSink
from receipt_intelligence.extraction.config import ExtractionRequest
from receipt_intelligence.observability.timing import utc_now_iso

OPENAI_RECEIPT_OPERATION = "receipt_extraction_one_shot"


class ObservedOpenAIClient:
    """Decorate an OpenAI SDK client and publish one event per Responses call."""

    def __init__(
        self,
        delegate: Any,
        event_sink: EventSink,
        *,
        run_id: str,
        default_model: str,
    ) -> None:
        self._delegate = delegate
        self.responses = _ObservedResponsesResource(
            delegate.responses,
            event_sink,
            run_id=run_id,
            default_model=default_model,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _ObservedResponsesResource:
    def __init__(
        self,
        delegate: Any,
        event_sink: EventSink,
        *,
        run_id: str,
        default_model: str,
    ) -> None:
        self._delegate = delegate
        self._event_sink = event_sink
        self._run_id = str(run_id)
        self._default_model = str(default_model)

    def create(self, **kwargs: Any) -> Any:
        call_id = f"mc_{uuid.uuid4().hex}"
        started_at = utc_now_iso()
        started = time.perf_counter()
        model = str(kwargs.get("model") or self._default_model)
        input_characters, request_attributes = _request_attributes(kwargs)

        try:
            response = self._delegate.create(**kwargs)
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            _publish_safely(
                self._event_sink,
                ModelCallCompletedEvent(
                    call_id=call_id,
                    occurred_at=utc_now_iso(),
                    started_at=started_at,
                    trace_id=self._run_id,
                    job_id=self._run_id,
                    operation=OPENAI_RECEIPT_OPERATION,
                    provider="openai",
                    model=model,
                    endpoint="generate",
                    status="failed",
                    attempt=1,
                    duration_ms=duration_ms,
                    input_characters=input_characters,
                    output_characters=0,
                    token_source="unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                    attributes=request_attributes,
                ),
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000.0
        usage = _usage_breakdown(response)
        response_status = _optional_text(getattr(response, "status", None))
        event_status = "completed" if response_status in {None, "completed"} else "failed"
        output_text = str(getattr(response, "output_text", "") or "")
        attributes = {
            **request_attributes,
            "response_id": _optional_text(getattr(response, "id", None)),
            "response_status": response_status,
            "service_tier": _optional_text(getattr(response, "service_tier", None)),
            "total_tokens": usage["total_tokens"],
            "cached_input_tokens": usage["cached_input_tokens"],
            "uncached_input_tokens": usage["uncached_input_tokens"],
            "cache_write_input_tokens": usage["cache_write_input_tokens"],
            "reasoning_output_tokens": usage["reasoning_output_tokens"],
        }
        attributes = {key: value for key, value in attributes.items() if value is not None}
        response_model = _optional_text(getattr(response, "model", None)) or model
        error = (
            None
            if event_status == "completed"
            else f"OpenAI response status was {response_status!r}."
        )
        _publish_safely(
            self._event_sink,
            ModelCallCompletedEvent(
                call_id=call_id,
                occurred_at=utc_now_iso(),
                started_at=started_at,
                trace_id=self._run_id,
                job_id=self._run_id,
                operation=OPENAI_RECEIPT_OPERATION,
                provider="openai",
                model=response_model,
                endpoint="generate",
                status=event_status,
                attempt=1,
                duration_ms=duration_ms,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                input_characters=input_characters,
                output_characters=len(output_text),
                token_source="provider_reported",
                stop_reason=response_status,
                error=error,
                attributes=attributes,
            ),
        )
        return response


def build_observed_openai_client(
    config: ExtractionRequest,
    *,
    client: Any | None = None,
    event_sink: EventSink | None = None,
) -> ObservedOpenAIClient:
    """Build the OpenAI SDK client decorated with the app's telemetry sink."""

    if client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set for the OpenAI extraction backend.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - operational dependency failure
            raise RuntimeError(
                "OpenAI backend requires the 'openai' package. Rebuild/install requirements/app.txt."
            ) from exc
        client = OpenAI(timeout=config.openai_timeout_seconds)

    if event_sink is None:
        from receipt_intelligence import settings

        event_sink = SQLiteModelCallRepository(
            settings.RECEIPT_DB_PATH,
            enabled=settings.MODEL_CALL_TELEMETRY_ENABLED,
        )

    return ObservedOpenAIClient(
        client,
        event_sink,
        run_id=config.run_id,
        default_model=config.openai_model,
    )


def publish_openai_extraction_metrics(
    config: ExtractionRequest,
    *,
    status: str,
    started_at: str,
    duration_ms: float,
    stages: Sequence[Mapping[str, Any]],
    error: str | None = None,
) -> Any:
    """Persist the same extraction metrics artifact used by the local backend."""

    path = config.result_dir / f"{config.run_id}_extraction_metrics.json"
    sink = JsonFileEventSink(
        path,
        aliases=(config.result_dir / "latest_extraction_metrics.json",),
    )
    sink.publish(
        ExtractionRunEvent(
            run_id=config.run_id,
            status=status,
            started_at=started_at,
            occurred_at=utc_now_iso(),
            duration_ms=max(0.0, float(duration_ms)),
            stages=tuple(dict(stage) for stage in stages),
            error=error,
        )
    )
    return path


def _request_attributes(kwargs: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    input_characters = len(str(kwargs.get("instructions") or ""))
    image_count = 0
    image_details: list[str] = []

    input_value = kwargs.get("input")
    if isinstance(input_value, str):
        input_characters += len(input_value)
    elif isinstance(input_value, list):
        for message in input_value:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if isinstance(content, str):
                input_characters += len(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                part_type = str(part.get("type") or "")
                if part_type == "input_text":
                    input_characters += len(str(part.get("text") or ""))
                elif part_type == "input_image":
                    image_count += 1
                    detail = _optional_text(part.get("detail"))
                    if detail:
                        image_details.append(detail)

    reasoning = kwargs.get("reasoning")
    reasoning_effort = (
        _optional_text(reasoning.get("effort")) if isinstance(reasoning, Mapping) else None
    )
    attributes: dict[str, Any] = {
        "modality": "multimodal" if image_count else "text",
        "image_count": image_count,
        "image_detail": image_details[0] if len(set(image_details)) == 1 else image_details or None,
        "reasoning_effort": reasoning_effort or "none",
        "store": kwargs.get("store"),
        "max_output_tokens": _optional_int(kwargs.get("max_output_tokens")),
        "structured_output": bool(
            isinstance(kwargs.get("text"), Mapping)
            and isinstance(kwargs.get("text", {}).get("format"), Mapping)
        ),
    }
    return input_characters, {key: value for key, value in attributes.items() if value is not None}


def _usage_breakdown(response: Any) -> dict[str, int | None]:
    usage = _as_mapping(getattr(response, "usage", None))
    input_tokens = _optional_int(usage.get("input_tokens"))
    output_tokens = _optional_int(usage.get("output_tokens"))
    total_tokens = _optional_int(usage.get("total_tokens"))

    input_details = _as_mapping(
        usage.get("input_tokens_details") or usage.get("input_token_details")
    )
    output_details = _as_mapping(
        usage.get("output_tokens_details") or usage.get("output_token_details")
    )
    cached_input_tokens = _first_int(
        input_details.get("cached_tokens"),
        usage.get("input_cached_tokens"),
        usage.get("cached_tokens"),
    )
    cache_write_input_tokens = _first_int(
        input_details.get("cache_write_tokens"),
        input_details.get("cache_creation_tokens"),
        usage.get("cache_write_tokens"),
        usage.get("cache_creation_tokens"),
    )
    reasoning_output_tokens = _first_int(
        output_details.get("reasoning_tokens"),
        usage.get("reasoning_tokens"),
    )
    uncached_input_tokens = (
        max(0, input_tokens - cached_input_tokens)
        if input_tokens is not None and cached_input_tokens is not None
        else None
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
    }


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _publish_safely(event_sink: EventSink, event: ModelCallCompletedEvent) -> None:
    try:
        event_sink.publish(event)
    except Exception:
        # Telemetry is observational and must never alter model-call behavior.
        return


__all__ = [
    "OPENAI_RECEIPT_OPERATION",
    "ObservedOpenAIClient",
    "build_observed_openai_client",
    "publish_openai_extraction_metrics",
]
