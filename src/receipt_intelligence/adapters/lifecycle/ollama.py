"""Ollama accelerator lifecycle adapter."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.application.ports.model_lifecycle import (
    ModelLifecycleCoordinator,
    ModelLifecycleRequest,
)
from receipt_intelligence.services.ollama_control import reload_ollama, unload_ollama


class OllamaModelLifecycleCoordinator(ModelLifecycleCoordinator):
    def __init__(
        self,
        *,
        base_url: str,
        control_mode: str = "api",
        unload_command: str = "",
        start_command: str = "",
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.control_mode = str(control_mode or "api").strip().lower()
        self.unload_command = str(unload_command or "")
        self.start_command = str(start_command or "")

    def release_for_vlm(self, request: ModelLifecycleRequest) -> dict[str, Any]:
        return unload_ollama(
            ollama_url=self.base_url,
            model=request.model,
            timeout_seconds=request.timeout_seconds,
            mode=self.control_mode,
            unload_command=self.unload_command,
            wait_seconds=request.wait_seconds,
        )

    def restore_after_vlm(self, request: ModelLifecycleRequest) -> dict[str, Any]:
        return reload_ollama(
            ollama_url=self.base_url,
            model=request.model,
            keep_alive=request.keep_alive or "10m",
            timeout_seconds=request.timeout_seconds,
            mode=self.control_mode,
            start_command=self.start_command,
            warmup_prompt=request.warmup_prompt,
            wait_seconds=request.wait_seconds,
        )


__all__ = ["OllamaModelLifecycleCoordinator"]
