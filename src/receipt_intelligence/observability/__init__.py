"""Observability public API with dependency-safe lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from receipt_intelligence.application.ports.llm import ModelCallMetrics as ModelCallMetrics
    from receipt_intelligence.observability.jsonl import JsonlEventWriter as JsonlEventWriter
    from receipt_intelligence.observability.query import QueryTelemetrySink as QueryTelemetrySink
    from receipt_intelligence.observability.readiness import (
        build_readiness_report as build_readiness_report,
    )
    from receipt_intelligence.observability.timing import elapsed_ms as elapsed_ms
    from receipt_intelligence.observability.timing import utc_now_iso as utc_now_iso

_EXPORTS: dict[str, tuple[str, str]] = {
    "JsonlEventWriter": ("receipt_intelligence.observability.jsonl", "JsonlEventWriter"),
    "ModelCallMetrics": ("receipt_intelligence.application.ports.llm", "ModelCallMetrics"),
    "QueryTelemetrySink": (
        "receipt_intelligence.observability.query",
        "QueryTelemetrySink",
    ),
    "build_readiness_report": (
        "receipt_intelligence.observability.readiness",
        "build_readiness_report",
    ),
    "elapsed_ms": ("receipt_intelligence.observability.timing", "elapsed_ms"),
    "utc_now_iso": ("receipt_intelligence.observability.timing", "utc_now_iso"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve one observability symbol without importing readiness at package load."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
