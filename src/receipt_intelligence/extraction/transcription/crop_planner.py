"""Aspect-ratio-adaptive, Paddle-snapped, pixel-verified receipt crop planning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from PIL import Image, ImageOps

from receipt_intelligence.extraction.settings import CropPlanningSettings
from receipt_intelligence.extraction.transcription.models import (
    CropPlan,
    CropSpec,
    DetectedLine,
    VerifiedCutBoundary,
)


def _horizontal_text_roi(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    horizontal_padding: int,
    full_width: bool,
) -> tuple[int, int]:
    image_width, _ = image.size
    if full_width or not lines:
        return 0, image_width
    left = max(0, math.floor(min(line.x_min for line in lines) - horizontal_padding))
    right = min(image_width, math.ceil(max(line.x_max for line in lines) + horizontal_padding))
    return (0, image_width) if right <= left else (left, right)


def _strip_ink_density(
    grayscale: Image.Image,
    *,
    y: int,
    left: int,
    right: int,
    half_height: int,
    dark_threshold: int,
) -> tuple[float, int, int]:
    image_width, image_height = grayscale.size
    left = max(0, min(image_width - 1, left))
    right = max(left + 1, min(image_width, right))
    strip_top = max(0, y - max(0, half_height))
    strip_bottom = min(image_height, y + max(0, half_height) + 1)
    strip = grayscale.crop((left, strip_top, right, strip_bottom))
    histogram = strip.histogram()
    threshold = max(0, min(255, int(dark_threshold)))
    dark_pixels = sum(histogram[: threshold + 1])
    total_pixels = max(1, strip.width * strip.height)
    return dark_pixels / total_pixels, strip_top, strip_bottom


def _find_verified_boundary(
    image: Image.Image,
    grayscale: Image.Image,
    lines: Sequence[DetectedLine],
    cut_index: int,
    *,
    roi_left: int,
    roi_right: int,
    settings: CropPlanningSettings,
) -> VerifiedCutBoundary | None:
    if cut_index <= 0:
        return VerifiedCutBoundary(0, 0, float("inf"), 0.0, 0, 0, roi_left, roi_right)
    if cut_index >= len(lines):
        return VerifiedCutBoundary(
            len(lines),
            image.height,
            float("inf"),
            0.0,
            image.height,
            image.height,
            roi_left,
            roi_right,
        )

    previous = lines[cut_index - 1]
    following = lines[cut_index]
    geometric_gap = float(following.y_min - previous.y_max)
    if geometric_gap < settings.minimum_safe_gap:
        return None

    margin = max(0, int(settings.cut_search_margin))
    start_y = max(1, math.ceil(previous.y_max) + margin)
    end_y = min(image.height - 1, math.floor(following.y_min) - margin)
    if end_y < start_y:
        return None

    midpoint = (previous.y_max + following.y_min) / 2.0
    candidates: list[tuple[float, float, int, int, int]] = []
    for y in range(start_y, end_y + 1):
        density, strip_top, strip_bottom = _strip_ink_density(
            grayscale,
            y=y,
            left=roi_left,
            right=roi_right,
            half_height=settings.cut_strip_half_height,
            dark_threshold=settings.cut_ink_threshold,
        )
        candidates.append((density, abs(float(y) - midpoint), y, strip_top, strip_bottom))
    if not candidates:
        return None
    density, _, y, strip_top, strip_bottom = min(candidates)
    if density > settings.maximum_cut_ink_density:
        return None
    return VerifiedCutBoundary(
        cut_index=cut_index,
        y=y,
        geometric_gap_pixels=geometric_gap,
        ink_density=density,
        strip_top=strip_top,
        strip_bottom=strip_bottom,
        roi_left=roi_left,
        roi_right=roi_right,
    )


def _precompute_boundaries(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    settings: CropPlanningSettings,
) -> dict[int, VerifiedCutBoundary]:
    grayscale = ImageOps.grayscale(image)
    roi_left, roi_right = _horizontal_text_roi(
        image,
        lines,
        settings.horizontal_padding,
        settings.full_width_crops,
    )
    boundaries: dict[int, VerifiedCutBoundary] = {}
    for cut_index in range(len(lines) + 1):
        boundary = _find_verified_boundary(
            image,
            grayscale,
            lines,
            cut_index,
            roi_left=roi_left,
            roi_right=roi_right,
            settings=settings,
        )
        if boundary is not None:
            boundaries[cut_index] = boundary
    return boundaries


def determine_effective_crop_count(
    *,
    requested_crops: int,
    detected_line_count: int,
    image_width: int,
    image_height: int,
    target_rows_per_crop: int,
    single_crop_max_rows: int,
    single_crop_max_aspect_ratio: float,
) -> tuple[int, dict[str, Any]]:
    if requested_crops < 1 or target_rows_per_crop < 1:
        raise ValueError("Crop and row targets must be positive.")
    if single_crop_max_rows < 1 or single_crop_max_aspect_ratio <= 0:
        raise ValueError("Single-crop thresholds must be positive.")
    if image_width < 1 or image_height < 1:
        raise ValueError("Image dimensions must be positive.")

    aspect_ratio = float(image_height) / float(image_width)
    small_by_rows = detected_line_count <= single_crop_max_rows
    small_by_aspect_ratio = aspect_ratio <= single_crop_max_aspect_ratio
    if requested_crops == 1 or (small_by_rows and small_by_aspect_ratio):
        effective = 1
        reason = "requested_single_crop" if requested_crops == 1 else "small_receipt"
    else:
        row_based = max(1, math.ceil(detected_line_count / target_rows_per_crop))
        aspect_based = max(1, math.ceil(aspect_ratio / single_crop_max_aspect_ratio))
        effective = min(requested_crops, max(2, row_based, aspect_based))
        reason = "adaptive_up_to_requested_crop_count"
    return effective, {
        "requested_crops": requested_crops,
        "initial_effective_crops": effective,
        "reason": reason,
        "detected_line_count_estimate": detected_line_count,
        "image_width": image_width,
        "image_height": image_height,
        "image_aspect_ratio_h_over_w": round(aspect_ratio, 6),
        "target_rows_per_crop": target_rows_per_crop,
        "single_crop_max_rows": single_crop_max_rows,
        "single_crop_max_aspect_ratio": single_crop_max_aspect_ratio,
        "small_by_rows": small_by_rows,
        "small_by_aspect_ratio": small_by_aspect_ratio,
    }


def _build_crop(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    line_indices: Sequence[int],
    *,
    crop_id: str,
    top: int,
    bottom: int,
    settings: CropPlanningSettings,
    full_image_fallback: bool = False,
) -> CropSpec:
    if not line_indices and not full_image_fallback:
        raise ValueError("Cannot build an empty safe-cut crop.")
    ordered = tuple(sorted(line_indices))
    if ordered and tuple(range(ordered[0], ordered[-1] + 1)) != ordered:
        raise ValueError("Safe-cut crops must contain contiguous detected rows.")
    top = max(0, min(image.height - 1, int(top)))
    bottom = max(top + 1, min(image.height, int(bottom)))
    left, right = _horizontal_text_roi(
        image,
        lines,
        settings.horizontal_padding,
        settings.full_width_crops,
    )
    return CropSpec(
        crop_id=crop_id,
        line_indices=ordered,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        image=image.crop((left, top, right, bottom)),
        is_full_image_fallback=full_image_fallback,
    )


def full_image_crop(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    *,
    crop_id: str = "GFULL",
    fallback: bool = True,
) -> CropSpec:
    return CropSpec(
        crop_id=crop_id,
        line_indices=tuple(range(len(lines))),
        top=0,
        bottom=image.height,
        left=0,
        right=image.width,
        image=image.copy(),
        is_full_image_fallback=fallback,
    )


def _select_boundary(
    boundaries: dict[int, VerifiedCutBoundary],
    *,
    nominal_y: float,
    previous_cut_index: int,
    detected_line_count: int,
    remaining_crops_after_cut: int,
    settings: CropPlanningSettings,
    normal_radius: float,
    maximum_radius: float,
) -> tuple[VerifiedCutBoundary | None, str]:
    minimum_cut_index = previous_cut_index + settings.minimum_lines_per_crop
    maximum_cut_index = (
        detected_line_count - remaining_crops_after_cut * settings.minimum_lines_per_crop
    )
    if minimum_cut_index > maximum_cut_index:
        return None, "insufficient_detected_lines_for_remaining_crops"
    candidates = [
        boundary
        for cut_index, boundary in boundaries.items()
        if minimum_cut_index <= cut_index <= maximum_cut_index
        and 0 < cut_index < detected_line_count
    ]
    if not candidates:
        return None, "no_verified_boundary_in_allowed_line_range"
    normal = [
        boundary for boundary in candidates if abs(float(boundary.y) - nominal_y) <= normal_radius
    ]
    if normal:
        pool = normal
        stage = "normal_search_radius"
    else:
        pool = [
            boundary
            for boundary in candidates
            if abs(float(boundary.y) - nominal_y) <= maximum_radius
        ]
        if not pool:
            return None, "no_verified_boundary_within_max_search_radius"
        stage = "expanded_search_radius"
    selected = min(
        pool,
        key=lambda boundary: (
            abs(float(boundary.y) - nominal_y),
            boundary.ink_density,
            -boundary.geometric_gap_pixels,
            boundary.cut_index,
        ),
    )
    return selected, stage


def _try_plan(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    boundaries: dict[int, VerifiedCutBoundary],
    *,
    crop_count: int,
    settings: CropPlanningSettings,
) -> tuple[list[CropSpec] | None, list[dict[str, Any]], str | None]:
    if crop_count < 1:
        return None, [], "crop_count_below_one"
    if 0 not in boundaries or len(lines) not in boundaries:
        return None, [], "missing_image_edge_boundaries"
    if crop_count == 1:
        return (
            [full_image_crop(image, lines, fallback=False)],
            [{"boundary_number": 0, "selection": "single_full_image_crop"}],
            None,
        )
    if len(lines) < crop_count * settings.minimum_lines_per_crop:
        return None, [], "insufficient_detected_lines_for_minimum_lines_per_crop"

    nominal_crop_height = float(image.height) / float(crop_count)
    normal_radius = max(1.0, nominal_crop_height * settings.safe_cut_search_ratio)
    maximum_radius = max(
        normal_radius,
        nominal_crop_height * settings.maximum_safe_cut_search_ratio,
    )
    selected_boundaries = [boundaries[0]]
    decisions: list[dict[str, Any]] = []
    previous_cut_index = 0
    for boundary_number in range(1, crop_count):
        nominal_y = float(image.height) * boundary_number / crop_count
        selected, stage = _select_boundary(
            boundaries,
            nominal_y=nominal_y,
            previous_cut_index=previous_cut_index,
            detected_line_count=len(lines),
            remaining_crops_after_cut=crop_count - boundary_number,
            settings=settings,
            normal_radius=normal_radius,
            maximum_radius=maximum_radius,
        )
        if selected is None:
            return None, decisions, f"boundary_{boundary_number}_{stage}"
        if selected.y <= selected_boundaries[-1].y:
            return None, decisions, f"boundary_{boundary_number}_not_monotonic"
        decisions.append(
            {
                "boundary_number": boundary_number,
                "nominal_y": round(nominal_y, 3),
                "selected_y": selected.y,
                "distance_from_nominal_pixels": round(abs(selected.y - nominal_y), 3),
                "search_stage": stage,
                "cut_index": selected.cut_index,
                "ink_density": selected.ink_density,
                "geometric_gap_pixels": selected.geometric_gap_pixels,
                "strip_top": selected.strip_top,
                "strip_bottom": selected.strip_bottom,
            }
        )
        selected_boundaries.append(selected)
        previous_cut_index = selected.cut_index
    selected_boundaries.append(boundaries[len(lines)])

    crops: list[CropSpec] = []
    for crop_index in range(crop_count):
        top_boundary = selected_boundaries[crop_index]
        bottom_boundary = selected_boundaries[crop_index + 1]
        if bottom_boundary.cut_index <= top_boundary.cut_index:
            return None, decisions, f"crop_{crop_index + 1}_empty_line_range"
        crops.append(
            _build_crop(
                image,
                lines,
                range(top_boundary.cut_index, bottom_boundary.cut_index),
                crop_id=f"G{crop_index + 1:03d}",
                top=top_boundary.y,
                bottom=bottom_boundary.y,
                settings=settings,
            )
        )
    return crops, decisions, None


def plan_safe_crops(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    settings: CropPlanningSettings,
) -> CropPlan:
    if not lines:
        raise ValueError("Safe crop planning requires detected lines.")
    boundaries = _precompute_boundaries(image, lines, settings)
    if 0 not in boundaries or len(lines) not in boundaries:
        raise RuntimeError("Image-edge safe boundaries could not be established.")
    desired_count, metadata = determine_effective_crop_count(
        requested_crops=settings.max_crops,
        detected_line_count=len(lines),
        image_width=image.width,
        image_height=image.height,
        target_rows_per_crop=settings.target_rows_per_crop,
        single_crop_max_rows=settings.single_crop_max_rows,
        single_crop_max_aspect_ratio=settings.single_crop_max_aspect_ratio,
    )
    attempts: list[dict[str, Any]] = []
    for crop_count in range(desired_count, 0, -1):
        crops, decisions, failure = _try_plan(
            image,
            lines,
            boundaries,
            crop_count=crop_count,
            settings=settings,
        )
        attempts.append(
            {
                "crop_count": crop_count,
                "status": "planned" if crops is not None else "rejected",
                "failure": failure,
            }
        )
        if crops is not None:
            return CropPlan(
                crops=tuple(crops),
                boundaries=tuple(boundaries.values()),
                boundary_decisions=tuple(decisions),
                metadata={
                    **metadata,
                    "effective_crops": crop_count,
                    "crop_count_reduced_for_safety": crop_count < desired_count,
                    "status": "planned",
                    "verified_boundary_count": len(boundaries),
                    "attempts": attempts,
                },
            )
    raise RuntimeError("Could not create even a single complete Qwen crop plan.")


__all__ = ["determine_effective_crop_count", "full_image_crop", "plan_safe_crops"]
