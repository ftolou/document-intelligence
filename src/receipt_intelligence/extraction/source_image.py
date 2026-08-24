"""Bounded generic validation for source images before extraction."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

_EXIF_ORIENTATION_TAG = 274
_TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})


class SourceImageValidationError(ValueError):
    """Raised when a source image fails the generic Core input contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_source_image(
    source_image_path: str | Path,
    *,
    max_width: int,
    max_height: int,
    max_pixels: int,
) -> None:
    """Validate one image using caller-supplied resource bounds before extraction.

    The check is intentionally provider-independent: Pillow is used only as the
    decoder boundary. File-format allowlisting and upload policy remain outside
    this contract.
    """

    if max_width < 1 or max_height < 1 or max_pixels < 1:
        raise ValueError("Source image validation limits must be positive.")

    path = Path(source_image_path)
    if not path.is_file():
        raise SourceImageValidationError(
            "source_image_not_found",
            "Source image does not exist or is not a regular file.",
        )

    from PIL import Image, UnidentifiedImageError

    try:
        with warnings.catch_warnings():
            # Apply the explicit caller limit below instead of relying on Pillow's
            # warning threshold, while retaining Pillow's hard decoder ceiling.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)

            with Image.open(path) as image:
                _validate_raw_dimensions(
                    image,
                    max_width=max_width,
                    max_height=max_height,
                    max_pixels=max_pixels,
                )
                image.verify()

            # verify() intentionally invalidates the decoder, so reopen and force
            # a complete decode after the cheap bounded header check.
            with Image.open(path) as image:
                _validate_raw_dimensions(
                    image,
                    max_width=max_width,
                    max_height=max_height,
                    max_pixels=max_pixels,
                )
                image.load()
                _validate_oriented_dimensions(
                    image,
                    max_width=max_width,
                    max_height=max_height,
                )
    except SourceImageValidationError:
        raise
    except Image.DecompressionBombError as exc:
        raise SourceImageValidationError(
            "source_image_resource_limit_exceeded",
            "Source image exceeds the decoder safety limit.",
        ) from exc
    except (UnidentifiedImageError, EOFError, OSError, SyntaxError, ValueError) as exc:
        raise SourceImageValidationError(
            "source_image_malformed",
            "Source image is malformed, corrupted, or truncated.",
        ) from exc


def _validate_raw_dimensions(
    image: Any,
    *,
    max_width: int,
    max_height: int,
    max_pixels: int,
) -> None:
    raw_width, raw_height = image.size
    if raw_width < 1 or raw_height < 1:
        raise SourceImageValidationError(
            "source_image_invalid_dimensions",
            "Source image has invalid dimensions.",
        )

    if raw_width * raw_height > max_pixels:
        raise SourceImageValidationError(
            "source_image_pixel_limit_exceeded",
            "Source image pixel count exceeds the configured limit.",
        )

    # Reject before decoding when no EXIF orientation could make a raw side fit
    # either configured effective dimension.
    max_side = max(max_width, max_height)
    if raw_width > max_side or raw_height > max_side:
        raise SourceImageValidationError(
            "source_image_dimension_limit_exceeded",
            "Source image dimensions exceed the configured limits.",
        )


def _validate_oriented_dimensions(
    image: Any,
    *,
    max_width: int,
    max_height: int,
) -> None:
    raw_width, raw_height = image.size
    orientation = image.getexif().get(_EXIF_ORIENTATION_TAG)
    width, height = (
        (raw_height, raw_width)
        if orientation in _TRANSPOSED_ORIENTATIONS
        else (raw_width, raw_height)
    )
    if width > max_width or height > max_height:
        raise SourceImageValidationError(
            "source_image_dimension_limit_exceeded",
            "Source image dimensions exceed the configured limits.",
        )


__all__ = ["SourceImageValidationError", "validate_source_image"]
