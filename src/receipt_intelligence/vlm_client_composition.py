"""Main-application composition for the remote VLM service client only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from receipt_intelligence.adapters.vlm.remote_client import RemoteVlmClient
from receipt_intelligence.application.ports.vlm import VlmEngine
from receipt_intelligence.application.vlm.engines import OptionalVlmEngine, UnsupportedVlmEngine

if TYPE_CHECKING:
    from receipt_intelligence.extraction.config import ExtractionConfig

_REMOTE_BACKENDS = {
    "http_service",
    "http-service",
    "service",
    "vlm_service",
    # Legacy local names are routed across the HTTP boundary. The application
    # no longer hosts PaddleOCR-VL or invokes its CLI directly.
    "paddleocr_vl",
    "paddleocr-vl",
    "paddleocrvl",
    "local",
}


def build_client_vlm_engine(config: ExtractionConfig) -> VlmEngine:
    """Build the application-side client without importing VLM runtime code."""
    requested_backend = (config.vlm_backend or "http_service").strip().lower()
    if config.vlm_command.strip():
        delegate: VlmEngine = UnsupportedVlmEngine("command")
        effective_backend = "command"
    elif requested_backend in _REMOTE_BACKENDS:
        delegate = RemoteVlmClient(config.vlm_service_url)
        effective_backend = "http_service"
    else:
        delegate = UnsupportedVlmEngine(requested_backend)
        effective_backend = requested_backend

    return OptionalVlmEngine(delegate, backend_name=effective_backend)


__all__ = ["build_client_vlm_engine"]
