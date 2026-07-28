"""Provider-neutral visual document model contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class VlmRequest:
    image_path: Path | None
    result_dir: Path
    run_id: str
    timeout_seconds: float = 180.0
    progress_callback: ProgressCallback | None = None


class VlmEngine(Protocol):
    def analyze(self, request: VlmRequest) -> dict[str, Any]: ...


__all__ = ["ProgressCallback", "VlmEngine", "VlmRequest"]
