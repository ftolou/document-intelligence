from __future__ import annotations

from pathlib import Path

import pytest

from receipt_intelligence.application.ports import NullEventSink
from receipt_intelligence.extraction import ExtractionContext, ExtractionRequest
from receipt_intelligence.extraction.dependencies import ExtractionDependencies
from receipt_intelligence.extraction.state import (
    ExtractionPhase,
    PreparedArtifacts,
    StageContractError,
)


def _context(tmp_path: Path) -> ExtractionContext:
    return ExtractionContext(
        config=ExtractionRequest(
            source_image_path=tmp_path / "receipt.jpg",
            result_dir=tmp_path,
            run_id="state-contract",
            ollama_url="http://ollama",
            model="model",
        ),
        dependencies=ExtractionDependencies(
            llm_gateway=object(),  # type: ignore[arg-type]
            event_sink=NullEventSink(),
        ),
    )


def test_context_starts_without_partial_stage_artifacts(tmp_path: Path) -> None:
    context = _context(tmp_path)

    assert context.phase is ExtractionPhase.CREATED
    assert context.prepared is None
    assert context.transcription is None
    assert context.finalized is None
    with pytest.raises(StageContractError, match="prepared artifacts"):
        _ = context.paths


def test_transcription_state_requires_preparation(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(StageContractError, match="prepared artifacts"):
        context.begin_transcription_stage()

    context.prepared = PreparedArtifacts(paths={})
    artifacts = context.begin_transcription_stage()
    assert context.transcription is artifacts
