"""Convert provider-neutral Paddle regions into approximate physical receipt rows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from receipt_intelligence.application.ports.text_detection import DetectedTextRegion
from receipt_intelligence.extraction.settings import DetectionSettings
from receipt_intelligence.extraction.transcription.models import DetectedLine


def _bounds(region: DetectedTextRegion) -> tuple[float, float, float, float]:
    xs = [point[0] for point in region.polygon]
    ys = [point[1] for point in region.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _vertical_overlap_ratio(
    first_y_min: float,
    first_y_max: float,
    second_y_min: float,
    second_y_max: float,
) -> float:
    overlap = max(0.0, min(first_y_max, second_y_max) - max(first_y_min, second_y_min))
    denominator = max(
        1.0,
        min(first_y_max - first_y_min, second_y_max - second_y_min),
    )
    return overlap / denominator


def cluster_text_regions(
    regions: Sequence[DetectedTextRegion],
    settings: DetectionSettings,
) -> tuple[DetectedLine, ...]:
    accepted: list[tuple[DetectedTextRegion, float, float, float, float]] = []
    for region in regions:
        x_min, y_min, x_max, y_max = _bounds(region)
        if region.score is not None and region.score < settings.minimum_score:
            continue
        if (
            x_max - x_min < settings.minimum_box_width
            or y_max - y_min < settings.minimum_box_height
        ):
            continue
        accepted.append((region, x_min, y_min, x_max, y_max))

    working: list[dict[str, Any]] = []
    for region, x_min, y_min, x_max, y_max in sorted(
        accepted,
        key=lambda value: (((value[2] + value[4]) / 2.0), value[1]),
    ):
        center_y = (y_min + y_max) / 2.0
        box_height = max(1.0, y_max - y_min)
        best_index: int | None = None
        best_score = -1.0

        for index in range(max(0, len(working) - 4), len(working)):
            line = working[index]
            line_height = max(1.0, line["y_max"] - line["y_min"])
            overlap_ratio = _vertical_overlap_ratio(
                y_min,
                y_max,
                line["y_min"],
                line["y_max"],
            )
            center_delta = abs(center_y - ((line["y_min"] + line["y_max"]) / 2.0))
            center_limit = settings.line_center_factor * min(box_height, line_height)
            if overlap_ratio < settings.line_overlap_threshold and center_delta > center_limit:
                continue
            score = overlap_ratio - (center_delta / max(1.0, center_limit)) * 0.05
            if score > best_score:
                best_index = index
                best_score = score

        if best_index is None:
            working.append(
                {
                    "regions": [(region, x_min, y_min, x_max, y_max)],
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            )
            continue

        line = working[best_index]
        line["regions"].append((region, x_min, y_min, x_max, y_max))
        line["x_min"] = min(line["x_min"], x_min)
        line["y_min"] = min(line["y_min"], y_min)
        line["x_max"] = max(line["x_max"], x_max)
        line["y_max"] = max(line["y_max"], y_max)

    detected: list[DetectedLine] = []
    for line in sorted(
        working,
        key=lambda value: (((value["y_min"] + value["y_max"]) / 2.0), value["x_min"]),
    ):
        if (
            line["x_max"] - line["x_min"] < settings.minimum_line_width
            or line["y_max"] - line["y_min"] < settings.minimum_line_height
        ):
            continue
        members = sorted(line["regions"], key=lambda value: value[1])
        detected.append(
            DetectedLine(
                index=len(detected),
                region_ids=tuple(member[0].region_id for member in members),
                polygons=tuple(member[0].polygon for member in members),
                x_min=float(line["x_min"]),
                y_min=float(line["y_min"]),
                x_max=float(line["x_max"]),
                y_max=float(line["y_max"]),
            )
        )
    return tuple(detected)


__all__ = ["cluster_text_regions"]
