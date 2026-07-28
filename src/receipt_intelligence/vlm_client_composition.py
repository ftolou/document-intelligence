"""Main-application composition for the mandatory remote VLM service client."""

from __future__ import annotations

from typing import TYPE_CHECKING

from receipt_intelligence.adapters.vlm.remote_client import RemoteVlmClient
from receipt_intelligence.application.ports.vlm import VlmEngine
from receipt_intelligence.application.vlm.engines import RequiredVlmEngine, UnsupportedVlmEngine

if TYPE_CHECKING:
    from receipt_intelligence.extraction.config import ExtractionConfig

_REMOTE_BACKENDS = {
    "http_service",
    "http-service",
    "service",
    "vlm_service",
    # Historical local names are routed across the mandatory HTTP boundary.
    "paddleocr_vl",
    "paddleocr-vl",
    "paddleocrvl",
    "local",
}


def build_client_vlm_engine(config: ExtractionConfig) -> VlmEngine:
    """Build the application-side client without importing VLM runtime code."""
    requested_backend = (config.vlm_backend or "http_service").strip().lower()
    if requested_backend in _REMOTE_BACKENDS:
        delegate: VlmEngine = RemoteVlmClient(config.vlm_service_url)
        effective_backend = "http_service"
    else:
        delegate = UnsupportedVlmEngine(requested_backend)
        effective_backend = requested_backend

    return RequiredVlmEngine(delegate, backend_name=effective_backend)


__all__ = ["build_client_vlm_engine"]
