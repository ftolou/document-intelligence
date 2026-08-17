from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from receipt_intelligence.extraction.config import ExtractionRequest
from receipt_intelligence.extraction.openai_observability import (
    OPENAI_RECEIPT_OPERATION,
    ObservedOpenAIClient,
    publish_openai_extraction_metrics,
)


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def publish(self, event: Any) -> None:
        self.events.append(event)


class FakeResponses:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def _request_payload() -> dict[str, Any]:
    return {
        "model": "gpt-5.6-luna",
        "store": False,
        "max_output_tokens": 12000,
        "instructions": "system instructions",
        "reasoning": {"effort": "medium"},
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "extract receipt"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAAA",
                        "detail": "high",
                    },
                ],
            }
        ],
        "text": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
    }


def test_observed_openai_client_publishes_provider_usage_and_cache_breakdown() -> None:
    response = SimpleNamespace(
        id="resp_123",
        model="gpt-5.6-luna",
        status="completed",
        service_tier="default",
        output_text='{"ok":true}',
        usage={
            "input_tokens": 4790,
            "output_tokens": 3284,
            "total_tokens": 8074,
            "input_tokens_details": {"cached_tokens": 1395},
            "output_tokens_details": {"reasoning_tokens": 1588},
        },
    )
    sink = CollectingSink()
    responses = FakeResponses(response=response)
    client = ObservedOpenAIClient(
        FakeClient(responses),
        sink,
        run_id="job-openai-1",
        default_model="gpt-5.6-luna",
    )

    returned = client.responses.create(**_request_payload())

    assert returned is response
    assert len(responses.calls) == 1
    assert len(sink.events) == 1
    record = sink.events[0].to_record()
    assert record["operation"] == OPENAI_RECEIPT_OPERATION
    assert record["provider"] == "openai"
    assert record["model"] == "gpt-5.6-luna"
    assert record["job_id"] == "job-openai-1"
    assert record["trace_id"] == "job-openai-1"
    assert record["status"] == "completed"
    assert record["input_tokens"] == 4790
    assert record["output_tokens"] == 3284
    assert record["token_source"] == "provider_reported"
    assert record["attributes"]["cached_input_tokens"] == 1395
    assert record["attributes"]["uncached_input_tokens"] == 3395
    assert record["attributes"]["reasoning_output_tokens"] == 1588
    assert record["attributes"]["image_count"] == 1
    assert record["attributes"]["image_detail"] == "high"
    assert record["attributes"]["reasoning_effort"] == "medium"
    assert record["attributes"]["response_id"] == "resp_123"


def test_observed_openai_client_publishes_failed_call() -> None:
    sink = CollectingSink()
    client = ObservedOpenAIClient(
        FakeClient(FakeResponses(error=RuntimeError("provider unavailable"))),
        sink,
        run_id="job-openai-failed",
        default_model="gpt-5.6-luna",
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        client.responses.create(**_request_payload())

    assert len(sink.events) == 1
    record = sink.events[0].to_record()
    assert record["provider"] == "openai"
    assert record["status"] == "failed"
    assert record["job_id"] == "job-openai-failed"
    assert "provider unavailable" in str(record["error"])


def test_publish_openai_extraction_metrics_matches_local_artifact_contract(tmp_path: Path) -> None:
    request = ExtractionRequest(
        source_image_path=tmp_path / "receipt.png",
        result_dir=tmp_path,
        run_id="job-openai-metrics",
        ollama_url="",
        model="",
        extraction_backend="openai_one_shot",
        openai_model="gpt-5.6-luna",
    )

    path = publish_openai_extraction_metrics(
        request,
        status="completed",
        started_at="2026-08-17T10:00:00+00:00",
        duration_ms=1234.5,
        stages=(
            {"stage": "prepare", "status": "done", "duration_ms": 1.0},
            {"stage": "openai_one_shot", "status": "done", "duration_ms": 1200.0},
            {"stage": "validation", "status": "done", "duration_ms": 10.0},
        ),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["event_name"] == "extraction.run"
    assert payload["run_id"] == "job-openai-metrics"
    assert payload["status"] == "completed"
    assert payload["stage_count"] == 3
    assert payload["completed_stage_count"] == 3
    assert (tmp_path / "latest_extraction_metrics.json").exists()
