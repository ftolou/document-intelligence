"""Compatibility adapter around the existing VLM backend selector."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.application.ports.vlm import VlmEngine, VlmRequest
from receipt_intelligence.engines.vl_engine import run_optional_vlm


class ConfiguredVlmEngine(VlmEngine):
    def __init__(
        self,
        *,
        backend_name: str,
        service_url: str,
        trusted_command: str = "",
    ) -> None:
        self.backend_name = str(backend_name or "http_service").strip()
        self.service_url = str(service_url or "").strip()
        self.trusted_command = str(trusted_command or "")

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        return run_optional_vlm(
            image_path=request.image_path,
            result_dir=request.result_dir,
            run_id=request.run_id,
            enabled=request.enabled,
            backend=self.backend_name,
            service_url=self.service_url,
            command=self.trusted_command,
            timeout_seconds=request.timeout_seconds,
            progress_callback=request.progress_callback,
        )


__all__ = ["ConfiguredVlmEngine"]
