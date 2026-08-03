from __future__ import annotations

import pytest

from receipt_intelligence.extraction.strategy import (
    ExtractionStrategy,
    resolve_extraction_strategy,
    resolve_transcription_model,
)


def test_current_is_safe_default() -> None:
    assert resolve_extraction_strategy(environ={}) is ExtractionStrategy.CURRENT


def test_legacy_spatial_alias_remains_current() -> None:
    assert resolve_extraction_strategy("spatial_overview") is ExtractionStrategy.CURRENT
    assert (
        resolve_extraction_strategy(environ={"RECEIPT_EXTRACTION_STRATEGY": "spatial-overview"})
        is ExtractionStrategy.CURRENT
    )


def test_next_aliases_are_explicit() -> None:
    assert resolve_extraction_strategy("next") is ExtractionStrategy.NEXT
    assert resolve_extraction_strategy("qwen-gemma") is ExtractionStrategy.NEXT
    assert resolve_extraction_strategy(environ={"EXTRACTION_STRATEGY": "new"}) is (
        ExtractionStrategy.NEXT
    )


def test_invalid_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported extraction strategy"):
        resolve_extraction_strategy("automatic")


def test_transcription_model_environment() -> None:
    assert resolve_transcription_model(environ={}) == "qwen3.5:latest"
    assert (
        resolve_transcription_model(environ={"QWEN_TRANSCRIPTION_MODEL": "qwen3.5:9b"})
        == "qwen3.5:9b"
    )
