"""Accelerator/model lifecycle coordination contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelLifecycleRequest:
    model: str
    timeout_seconds: float = 120.0
    keep_alive: str | None = None
    warmup_prompt: str = "ok"
    wait_seconds: float = 0.0


class ModelLifecycleCoordinator(Protocol):
    def release_for_vlm(self, request: ModelLifecycleRequest) -> dict[str, Any]: ...

    def restore_after_vlm(self, request: ModelLifecycleRequest) -> dict[str, Any]: ...


class NoOpModelLifecycleCoordinator:
    def release_for_vlm(self, request: ModelLifecycleRequest) -> dict[str, Any]:
        return {"status": "skipped", "mode": "none", "model": request.model}

    def restore_after_vlm(self, request: ModelLifecycleRequest) -> dict[str, Any]:
        return {"status": "skipped", "mode": "none", "model": request.model}


__all__ = [
    "ModelLifecycleCoordinator",
    "ModelLifecycleRequest",
    "NoOpModelLifecycleCoordinator",
]
