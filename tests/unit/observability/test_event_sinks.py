from __future__ import annotations

import json
from pathlib import Path

from receipt_intelligence.adapters.observability import JsonFileEventSink, JsonlEventSink
from receipt_intelligence.application.events import ExtractionRunEvent


def _event() -> ExtractionRunEvent:
    return ExtractionRunEvent(
        run_id="receipt-1",
        status="completed",
        started_at="2026-07-24T12:00:00.000Z",
        occurred_at="2026-07-24T12:00:01.000Z",
        duration_ms=1000.0,
        stages=(
            {"stage": "prepare", "status": "done", "duration_ms": 100.0},
            {"stage": "parse", "status": "done", "duration_ms": 900.0},
        ),
    )


def test_json_file_sink_writes_event_and_alias(tmp_path: Path) -> None:
    path = tmp_path / "receipt-1_extraction_metrics.json"
    alias = tmp_path / "latest_extraction_metrics.json"

    JsonFileEventSink(path, aliases=(alias,)).publish(_event())

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == "extraction_metrics_v2"
    assert record["event_name"] == "extraction.run"
    assert record["completed_stage_count"] == 2
    assert json.loads(alias.read_text(encoding="utf-8")) == record


def test_jsonl_sink_is_disabled_without_creating_file(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"

    JsonlEventSink(path, enabled=False).publish(_event())

    assert not path.exists()
