"""Internal geometry values used by the transcription subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True, slots=True)
class DetectedLine:
    index: int
    region_ids: tuple[str, ...]
    polygons: tuple[tuple[tuple[float, float], ...], ...]
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0


@dataclass(frozen=True, slots=True)
class CropSpec:
    crop_id: str
    line_indices: tuple[int, ...]
    top: int
    bottom: int
    left: int
    right: int
    image: Image.Image
    is_full_image_fallback: bool = False


@dataclass(frozen=True, slots=True)
class VerifiedCutBoundary:
    cut_index: int
    y: int
    geometric_gap_pixels: float
    ink_density: float
    strip_top: int
    strip_bottom: int
    roi_left: int
    roi_right: int


@dataclass(frozen=True, slots=True)
class CropPlan:
    crops: tuple[CropSpec, ...]
    boundaries: tuple[VerifiedCutBoundary, ...]
    boundary_decisions: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


__all__ = ["CropPlan", "CropSpec", "DetectedLine", "VerifiedCutBoundary"]
