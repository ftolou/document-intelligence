"""Extraction workflow selection with a safe legacy default."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum


class ExtractionStrategy(StrEnum):
    CURRENT = "current"
    NEXT = "next"


_ALIASES = {
    "current": ExtractionStrategy.CURRENT,
    "legacy": ExtractionStrategy.CURRENT,
    "vlm": ExtractionStrategy.CURRENT,
    "default": ExtractionStrategy.CURRENT,
    "spatial_overview": ExtractionStrategy.CURRENT,
    "spatial-overview": ExtractionStrategy.CURRENT,
    "next": ExtractionStrategy.NEXT,
    "new": ExtractionStrategy.NEXT,
    "qwen_gemma": ExtractionStrategy.NEXT,
    "qwen-gemma": ExtractionStrategy.NEXT,
}


def resolve_extraction_strategy(
    value: str | ExtractionStrategy | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ExtractionStrategy:
    if isinstance(value, ExtractionStrategy):
        return value
    source = environ if environ is not None else os.environ
    configured = source.get("EXTRACTION_STRATEGY") or source.get(
        "RECEIPT_EXTRACTION_STRATEGY", "current"
    )
    raw = str(value if value is not None else configured)
    normalized = raw.strip().lower().replace(" ", "_")
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(strategy.value for strategy in ExtractionStrategy)
        raise ValueError(
            f"Unsupported extraction strategy {raw!r}; expected one of: {allowed}."
        ) from exc


def resolve_transcription_model(*, environ: Mapping[str, str] | None = None) -> str:
    source = environ if environ is not None else os.environ
    model = str(
        source.get("QWEN_TRANSCRIPTION_MODEL")
        or source.get("TRANSCRIPTION_MODEL")
        or "qwen3.5:latest"
    ).strip()
    if not model:
        raise ValueError("QWEN_TRANSCRIPTION_MODEL must not be empty.")
    return model


__all__ = [
    "ExtractionStrategy",
    "resolve_extraction_strategy",
    "resolve_transcription_model",
]
