"""Provider-neutral text-detection contracts for receipt crop planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class TextDetectionRequest:
    image_path: Path
    language: str = "german"
    device: str = "cpu"
    max_side_length: int | None = None

    def __post_init__(self) -> None:
        image_path = Path(self.image_path)
        language = str(self.language or "").strip()
        device = str(self.device or "").strip()
        if not language:
            raise ValueError("TextDetectionRequest.language must not be empty.")
        if not device:
            raise ValueError("TextDetectionRequest.device must not be empty.")
        if self.max_side_length is not None and self.max_side_length < 1:
            raise ValueError("max_side_length must be positive when provided.")
        object.__setattr__(self, "image_path", image_path)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "device", device)


@dataclass(frozen=True, slots=True)
class DetectedTextRegion:
    region_id: str
    polygon: Polygon
    score: float | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        region_id = str(self.region_id or "").strip()
        if not region_id:
            raise ValueError("DetectedTextRegion.region_id must not be empty.")
        if len(self.polygon) < 4:
            raise ValueError("DetectedTextRegion.polygon must contain at least four points.")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("DetectedTextRegion.score must be between 0 and 1.")
        object.__setattr__(self, "region_id", region_id)


@dataclass(frozen=True, slots=True)
class TextDetectionResult:
    regions: tuple[DetectedTextRegion, ...]
    image_width: int
    image_height: int
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("Detected image dimensions must be positive.")
        if self.duration_ms < 0:
            raise ValueError("TextDetectionResult.duration_ms must not be negative.")


class TextDetectionEngine(Protocol):
    def detect(self, request: TextDetectionRequest) -> TextDetectionResult: ...


__all__ = [
    "DetectedTextRegion",
    "Point",
    "Polygon",
    "TextDetectionEngine",
    "TextDetectionRequest",
    "TextDetectionResult",
]
