"""Bounded, semantic-free normalization of image and PDF document sources."""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from receipt_intelligence.extraction.source_image import (
    SourceImageValidationError,
    validate_source_image,
)

_PDF_SIGNATURE = b"%PDF-"
_EXIF_ORIENTATION_TAG = 274
_TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})
_SUPPORTED_IMAGE_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
}


class SourceNormalizationError(ValueError):
    """Raised when a document source cannot be normalized safely."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SourceNormalizationLimits:
    """Caller-supplied bounds applied before source pages are rasterized."""

    max_source_bytes: int
    max_pages: int
    max_page_width: int
    max_page_height: int
    max_page_pixels: int
    max_total_pixels: int

    def __post_init__(self) -> None:
        for field_name in (
            "max_source_bytes",
            "max_pages",
            "max_page_width",
            "max_page_height",
            "max_page_pixels",
            "max_total_pixels",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"SourceNormalizationLimits.{field_name} must be positive.")


@dataclass(frozen=True, slots=True)
class VisualPage:
    """One normalized visual page with a stable zero-based source index."""

    page_index: int
    image_bytes: bytes
    media_type: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.page_index < 0:
            raise ValueError("VisualPage.page_index must not be negative.")
        if not self.image_bytes:
            raise ValueError("VisualPage.image_bytes must not be empty.")
        if self.media_type != "image/png":
            raise ValueError("VisualPage.media_type must be image/png.")
        if self.width < 1 or self.height < 1:
            raise ValueError("VisualPage dimensions must be positive.")


@dataclass(frozen=True, slots=True)
class NormalizedDocumentSource:
    """A validated source represented as ordered, consistently encoded pages."""

    source_path: Path
    source_media_type: str
    pages: tuple[VisualPage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        if not self.source_media_type:
            raise ValueError("NormalizedDocumentSource.source_media_type must not be empty.")
        if not self.pages:
            raise ValueError("NormalizedDocumentSource.pages must not be empty.")
        if tuple(page.page_index for page in self.pages) != tuple(range(len(self.pages))):
            raise ValueError(
                "NormalizedDocumentSource page indices must be contiguous and ordered."
            )


def normalize_document_source(
    source_path: str | Path,
    *,
    limits: SourceNormalizationLimits,
    pdf_dpi: int = 144,
) -> NormalizedDocumentSource:
    """Normalize one JPEG, PNG, or PDF into bounded ordered PNG visual pages.

    The function performs only format validation and visual rasterization. It
    deliberately performs no OCR, text extraction, classification, or model
    invocation.
    """

    if pdf_dpi < 1:
        raise ValueError("pdf_dpi must be positive.")

    path = Path(source_path)
    if not path.is_file():
        raise SourceNormalizationError(
            "source_not_found",
            "Document source does not exist or is not a regular file.",
        )

    payload = _read_bounded_source(path, max_source_bytes=limits.max_source_bytes)
    if payload.startswith(_PDF_SIGNATURE):
        pages = _normalize_pdf(payload, limits=limits, pdf_dpi=pdf_dpi)
        source_media_type = "application/pdf"
    else:
        page, source_media_type = _normalize_image(path, payload=payload, limits=limits)
        pages = (page,)

    return NormalizedDocumentSource(
        source_path=path,
        source_media_type=source_media_type,
        pages=pages,
    )


def _read_bounded_source(path: Path, *, max_source_bytes: int) -> bytes:
    try:
        with path.open("rb") as source:
            payload = source.read(max_source_bytes + 1)
    except OSError as exc:
        raise SourceNormalizationError(
            "source_unreadable",
            "Document source could not be read.",
        ) from exc

    if len(payload) > max_source_bytes:
        raise SourceNormalizationError(
            "source_byte_limit_exceeded",
            "Document source exceeds the configured byte limit.",
        )
    if not payload:
        raise SourceNormalizationError("source_malformed", "Document source is empty.")
    return payload


def _normalize_image(
    path: Path,
    *,
    payload: bytes,
    limits: SourceNormalizationLimits,
) -> tuple[VisualPage, str]:
    try:
        validate_source_image(
            path,
            max_width=limits.max_page_width,
            max_height=limits.max_page_height,
            max_pixels=min(limits.max_page_pixels, limits.max_total_pixels),
        )
    except SourceImageValidationError as exc:
        pixel_limit_code = (
            "source_total_pixel_limit_exceeded"
            if limits.max_total_pixels < limits.max_page_pixels
            else "source_page_pixel_limit_exceeded"
        )
        code = {
            "source_image_dimension_limit_exceeded": "source_page_dimension_limit_exceeded",
            "source_image_pixel_limit_exceeded": pixel_limit_code,
            "source_image_resource_limit_exceeded": "source_resource_limit_exceeded",
        }.get(exc.code, "source_malformed")
        raise SourceNormalizationError(code, str(exc)) from exc

    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(BytesIO(payload)) as image:
            source_media_type = _SUPPORTED_IMAGE_FORMATS.get(image.format or "")
            if source_media_type is None:
                raise SourceNormalizationError(
                    "source_media_type_unsupported",
                    "Document source must be a JPEG, PNG, or PDF.",
                )
            raw_width, raw_height = image.size
            if raw_width * raw_height > limits.max_page_pixels:
                raise SourceNormalizationError(
                    "source_page_pixel_limit_exceeded",
                    "Document source exceeds the configured per-page pixel limit.",
                )
            orientation = image.getexif().get(_EXIF_ORIENTATION_TAG)
            width, height = (
                (raw_height, raw_width)
                if orientation in _TRANSPOSED_ORIENTATIONS
                else (raw_width, raw_height)
            )
            _validate_page_bounds(width, height, limits=limits)
            if width * height > limits.max_total_pixels:
                raise SourceNormalizationError(
                    "source_total_pixel_limit_exceeded",
                    "Document source exceeds the configured total pixel limit.",
                )
            image.load()
            normalized = ImageOps.exif_transpose(image).convert("RGBA")
            try:
                normalized.load()
                width, height = normalized.size
                _validate_page_bounds(width, height, limits=limits)
                if width * height > limits.max_total_pixels:
                    raise SourceNormalizationError(
                        "source_total_pixel_limit_exceeded",
                        "Document source exceeds the configured total pixel limit.",
                    )
                page = _visual_page(0, normalized)
            finally:
                normalized.close()
    except SourceNormalizationError:
        raise
    except (Image.DecompressionBombError, MemoryError) as exc:
        raise SourceNormalizationError(
            "source_resource_limit_exceeded",
            "Document source exceeds a decoder resource limit.",
        ) from exc
    except (UnidentifiedImageError, EOFError, OSError, SyntaxError, ValueError) as exc:
        raise SourceNormalizationError(
            "source_malformed",
            "Document source is malformed, corrupted, or truncated.",
        ) from exc

    return page, source_media_type


def _normalize_pdf(
    payload: bytes,
    *,
    limits: SourceNormalizationLimits,
    pdf_dpi: int,
) -> tuple[VisualPage, ...]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - exercised by package smoke tests
        raise SourceNormalizationError(
            "source_normalization_dependency_missing",
            "PDF normalization requires the source-normalization dependency extra.",
        ) from exc

    document = None
    try:
        document = pdfium.PdfDocument(payload)
        page_count = len(document)
        if page_count < 1:
            raise SourceNormalizationError("source_malformed", "PDF source has no pages.")
        if page_count > limits.max_pages:
            raise SourceNormalizationError(
                "source_page_limit_exceeded",
                "Document source exceeds the configured page limit.",
            )

        scale = pdf_dpi / 72.0
        expected_sizes = tuple(
            _pdf_page_size(document, page_index, scale=scale) for page_index in range(page_count)
        )
        total_pixels = 0
        for width, height in expected_sizes:
            _validate_page_bounds(width, height, limits=limits)
            total_pixels += width * height
            if total_pixels > limits.max_total_pixels:
                raise SourceNormalizationError(
                    "source_total_pixel_limit_exceeded",
                    "Document source exceeds the configured total pixel limit.",
                )

        rendered_pages: list[VisualPage] = []
        rendered_pixels = 0
        for page_index in range(page_count):
            page = _render_pdf_page(document, page_index, scale=scale, limits=limits)
            rendered_pixels += page.width * page.height
            if rendered_pixels > limits.max_total_pixels:
                raise SourceNormalizationError(
                    "source_total_pixel_limit_exceeded",
                    "Document source exceeds the configured total pixel limit.",
                )
            rendered_pages.append(page)
        pages = tuple(rendered_pages)
    except SourceNormalizationError:
        raise
    except (pdfium.PdfiumError, MemoryError) as exc:
        code = (
            "source_resource_limit_exceeded" if isinstance(exc, MemoryError) else "source_malformed"
        )
        raise SourceNormalizationError(
            code,
            "PDF source could not be decoded safely.",
        ) from exc
    except (OSError, OverflowError, ValueError) as exc:
        raise SourceNormalizationError(
            "source_malformed",
            "PDF source is malformed or has invalid page geometry.",
        ) from exc
    finally:
        if document is not None:
            document.close()

    return pages


def _pdf_page_size(document: object, page_index: int, *, scale: float) -> tuple[int, int]:
    page = document[page_index]  # type: ignore[index]
    try:
        page_width, page_height = page.get_size()
    finally:
        page.close()

    if (
        not math.isfinite(page_width)
        or not math.isfinite(page_height)
        or page_width <= 0
        or page_height <= 0
    ):
        raise SourceNormalizationError(
            "source_malformed",
            "PDF source has invalid page geometry.",
        )
    return math.ceil(page_width * scale), math.ceil(page_height * scale)


def _render_pdf_page(
    document: object,
    page_index: int,
    *,
    scale: float,
    limits: SourceNormalizationLimits,
) -> VisualPage:
    page = document[page_index]  # type: ignore[index]
    bitmap = None
    normalized = None
    try:
        bitmap = page.render(scale=scale)
        normalized = bitmap.to_pil().convert("RGBA")
        normalized.load()
        width, height = normalized.size
        _validate_page_bounds(width, height, limits=limits)
        return _visual_page(page_index, normalized)
    finally:
        if normalized is not None:
            normalized.close()
        if bitmap is not None:
            bitmap.close()
        page.close()


def _validate_page_bounds(
    width: int,
    height: int,
    *,
    limits: SourceNormalizationLimits,
) -> None:
    if width < 1 or height < 1:
        raise SourceNormalizationError(
            "source_malformed",
            "Document source has invalid page dimensions.",
        )
    if width > limits.max_page_width or height > limits.max_page_height:
        raise SourceNormalizationError(
            "source_page_dimension_limit_exceeded",
            "Document source exceeds the configured page dimension limits.",
        )
    if width * height > limits.max_page_pixels:
        raise SourceNormalizationError(
            "source_page_pixel_limit_exceeded",
            "Document source exceeds the configured per-page pixel limit.",
        )


def _visual_page(page_index: int, image: object) -> VisualPage:
    output = BytesIO()
    image.save(output, format="PNG")  # type: ignore[attr-defined]
    width, height = image.size  # type: ignore[attr-defined]
    return VisualPage(
        page_index=page_index,
        image_bytes=output.getvalue(),
        media_type="image/png",
        width=width,
        height=height,
    )


__all__ = [
    "NormalizedDocumentSource",
    "SourceNormalizationError",
    "SourceNormalizationLimits",
    "VisualPage",
    "normalize_document_source",
]
