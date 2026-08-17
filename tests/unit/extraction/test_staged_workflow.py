from __future__ import annotations

from pathlib import Path

import pytest

from receipt_intelligence.application.ports import NullEventSink
from receipt_intelligence.extraction import ExtractionContext, ExtractionRequest
from receipt_intelligence.extraction.dependencies import ExtractionDependencies
from receipt_intelligence.extraction.factory import build_extraction_workflow
from receipt_intelligence.extraction.workflow import ReceiptExtractionWorkflow


class _RecordingStage:
    def __init__(self, name: str, calls: list[str], *, fail: bool = False) -> None:
        self.name = name
        self.calls = calls
        self.fail = fail

    def run(self, context: ExtractionContext) -> ExtractionContext:
        self.calls.append(self.name)
        if self.fail:
            raise RuntimeError("stage failed")
        return context


def _context(tmp_path: Path) -> ExtractionContext:
    request = ExtractionRequest(
        source_image_path=tmp_path / "receipt.jpg",
        result_dir=tmp_path,
        run_id="test",
        ollama_url="http://ollama",
        model="model",
    )
    return ExtractionContext(
        config=request,
        dependencies=ExtractionDependencies(
            llm_gateway=object(),  # type: ignore[arg-type]
            event_sink=NullEventSink(),
        ),
    )


def test_canonical_workflow_has_explicit_ordered_stages() -> None:
    assert [stage.name for stage in build_extraction_workflow().stages] == [
        "prepare",
        "transcription",
        "structured_extraction",
        "validation",
        "correction",
        "categorization",
        "finalize",
    ]


def test_workflow_records_stage_order_and_stops_on_failure(tmp_path: Path) -> None:
    context = _context(tmp_path)
    calls: list[str] = []
    workflow = ReceiptExtractionWorkflow(
        [
            _RecordingStage("first", calls),
            _RecordingStage("second", calls, fail=True),
            _RecordingStage("third", calls),
        ]
    )

    with pytest.raises(RuntimeError, match="stage failed"):
        workflow.run(context)

    assert calls == ["first", "second"]
    assert context.stage_trace[0]["status"] == "done"
    assert context.stage_trace[1]["status"] == "error"
