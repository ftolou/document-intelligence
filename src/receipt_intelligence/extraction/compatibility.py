"""Strict compatibility mapping for historical extraction arguments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from receipt_intelligence.extraction.config import ExtractionRequest

LEGACY_EXTRACTION_ALIASES: dict[str, str] = {
    "vlm_gpu_orchestration": "gpu_orchestration",
    "gpu_orchestration_mode": "gpu_orchestration",
    "ollama_unload_before_vlm": "unload_llm_before_vlm",
    "unload_before_vlm": "unload_llm_before_vlm",
    "ollama_reload_after_vlm": "reload_llm_after_vlm",
    "reload_after_vlm": "reload_llm_after_vlm",
}

_EXTRACTION_FIELDS = frozenset(field.name for field in fields(ExtractionRequest))


def normalize_extraction_arguments(values: Mapping[str, Any]) -> dict[str, Any]:
    """Map supported historical names and reject unknown or duplicate fields."""

    normalized: dict[str, Any] = {}
    sources: dict[str, str] = {}
    unknown: list[str] = []

    for supplied_name, value in values.items():
        canonical_name = LEGACY_EXTRACTION_ALIASES.get(supplied_name, supplied_name)
        if canonical_name not in _EXTRACTION_FIELDS:
            unknown.append(supplied_name)
            continue
        if canonical_name in normalized:
            previous = sources[canonical_name]
            raise TypeError(
                "Extraction option supplied more than once: "
                f"{previous!r} and {supplied_name!r} both map to {canonical_name!r}"
            )
        normalized[canonical_name] = value
        sources[canonical_name] = supplied_name

    if unknown:
        names = ", ".join(sorted(repr(name) for name in unknown))
        raise TypeError(f"Unsupported extraction option(s): {names}")

    return normalized


def extraction_request_from_mapping(values: Mapping[str, Any]) -> ExtractionRequest:
    """Build a typed extraction request from canonical or supported legacy fields."""

    return ExtractionRequest(**normalize_extraction_arguments(values))


__all__ = [
    "LEGACY_EXTRACTION_ALIASES",
    "normalize_extraction_arguments",
    "extraction_request_from_mapping",
]
