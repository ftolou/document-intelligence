"""Provider-neutral OCR engine contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class OcrRequest:
    image_path: Path
    out_json_path: Path
    work_dir: Path
    lang: str = "german"
    device: str = "cpu"
    max_side_length: int = 4000
    detect_orientation: bool = True
    detection_max_side_length: int = 4000
    progress_callback: ProgressCallback | None = None


class OcrEngine(Protocol):
    def recognize(self, request: OcrRequest) -> dict[str, Any]: ...


__all__ = ["OcrEngine", "OcrRequest", "ProgressCallback"]
