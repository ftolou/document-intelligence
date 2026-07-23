"""Image preparation helpers for OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def prepare_image_for_ocr(
    image_path: Path, out_dir: Path, max_side_limit: int = 4000
) -> dict[str, Any]:
    """Normalize EXIF orientation and downscale if needed.

    PaddleOCR can warn/fail when the resized image exceeds its side limit. This
    helper makes that resize explicit and records it in OCR metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        original_width, original_height = img.size
        max_side = max(original_width, original_height)
        scale = 1.0
        if max_side_limit > 0 and max_side > max_side_limit:
            scale = max_side_limit / float(max_side)
            new_size = (
                max(1, int(round(original_width * scale))),
                max(1, int(round(original_height * scale))),
            )
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")
        prepared_path = out_dir / f"{image_path.stem}_ocr_input.jpg"
        img.save(prepared_path, quality=95)
        width, height = img.size
    return {
        "prepared_path": prepared_path,
        "original_width": original_width,
        "original_height": original_height,
        "image_width": width,
        "image_height": height,
        "scale": scale,
        "resized": scale != 1.0,
        "max_side_limit": max_side_limit,
    }
