from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from receipt_intelligence.extraction import (
    NormalizedDocumentSource,
    SourceNormalizationError,
    SourceNormalizationLimits,
    normalize_document_source,
)


def _limits(**overrides: int) -> SourceNormalizationLimits:
    values = {
        "max_source_bytes": 1_000_000,
        "max_pages": 5,
        "max_page_width": 100,
        "max_page_height": 100,
        "max_page_pixels": 10_000,
        "max_total_pixels": 30_000,
    }
    values.update(overrides)
    return SourceNormalizationLimits(**values)


def _write_image(path: Path, *, image_format: str, size: tuple[int, int] = (8, 6)) -> Path:
    Image.new("RGB", size, "white").save(path, format=image_format)
    return path


def _write_pdf(path: Path, colors: tuple[str, ...] = ("red", "green", "blue")) -> Path:
    pages = [Image.new("RGB", (12, 8), color) for color in colors]
    try:
        pages[0].save(path, format="PDF", save_all=True, append_images=pages[1:], resolution=72)
    finally:
        for page in pages:
            page.close()
    return path


@pytest.mark.parametrize(
    ("image_format", "suffix", "source_media_type"),
    [("JPEG", ".jpg", "image/jpeg"), ("PNG", ".png", "image/png")],
)
def test_image_becomes_one_normalized_visual_page(
    tmp_path: Path,
    image_format: str,
    suffix: str,
    source_media_type: str,
) -> None:
    path = _write_image(tmp_path / f"source{suffix}", image_format=image_format)

    source = normalize_document_source(path, limits=_limits())

    assert isinstance(source, NormalizedDocumentSource)
    assert source.source_path == path
    assert source.source_media_type == source_media_type
    assert len(source.pages) == 1
    page = source.pages[0]
    assert (page.page_index, page.media_type, page.width, page.height) == (
        0,
        "image/png",
        8,
        6,
    )
    with Image.open(BytesIO(page.image_bytes)) as normalized:
        assert normalized.format == "PNG"
        assert normalized.size == (8, 6)


def test_image_orientation_is_applied_before_page_dimensions_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (8, 6), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, format="JPEG", exif=exif)

    source = normalize_document_source(
        path,
        limits=_limits(max_page_width=6, max_page_height=8),
    )

    assert (source.pages[0].width, source.pages[0].height) == (6, 8)


def test_pdf_pages_keep_source_order_and_stable_indices(tmp_path: Path) -> None:
    path = _write_pdf(tmp_path / "three-pages.pdf")

    source = normalize_document_source(path, limits=_limits(), pdf_dpi=72)

    assert source.source_media_type == "application/pdf"
    assert [page.page_index for page in source.pages] == [0, 1, 2]
    dominant_channels: list[int] = []
    for page in source.pages:
        with Image.open(BytesIO(page.image_bytes)) as image:
            dominant_channels.append(max(range(3), key=image.getpixel((6, 4)).__getitem__))
    assert dominant_channels == [0, 1, 2]


@pytest.mark.parametrize(
    "field_name",
    [
        "max_source_bytes",
        "max_pages",
        "max_page_width",
        "max_page_height",
        "max_page_pixels",
        "max_total_pixels",
    ],
)
def test_limits_must_be_positive(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _limits(**{field_name: 0})


def test_source_byte_limit_is_checked_before_decoding(tmp_path: Path) -> None:
    path = tmp_path / "oversized.bin"
    path.write_bytes(b"not decoded" * 10)

    with pytest.raises(SourceNormalizationError) as exc_info:
        normalize_document_source(path, limits=_limits(max_source_bytes=10))

    assert exc_info.value.code == "source_byte_limit_exceeded"


def test_pdf_page_count_is_bounded(tmp_path: Path) -> None:
    path = _write_pdf(tmp_path / "three-pages.pdf")

    with pytest.raises(SourceNormalizationError) as exc_info:
        normalize_document_source(path, limits=_limits(max_pages=2), pdf_dpi=72)

    assert exc_info.value.code == "source_page_limit_exceeded"


@pytest.mark.parametrize(
    ("limits", "expected_code"),
    [
        ({"max_page_width": 11}, "source_page_dimension_limit_exceeded"),
        ({"max_page_pixels": 95}, "source_page_pixel_limit_exceeded"),
        ({"max_total_pixels": 200}, "source_total_pixel_limit_exceeded"),
    ],
)
def test_pdf_rasterization_is_bounded_before_pages_are_returned(
    tmp_path: Path,
    limits: dict[str, int],
    expected_code: str,
) -> None:
    path = _write_pdf(tmp_path / "three-pages.pdf")

    with pytest.raises(SourceNormalizationError) as exc_info:
        normalize_document_source(path, limits=_limits(**limits), pdf_dpi=72)

    assert exc_info.value.code == expected_code


def test_image_total_pixel_limit_is_applied(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "source.png", image_format="PNG")

    with pytest.raises(SourceNormalizationError) as exc_info:
        normalize_document_source(path, limits=_limits(max_total_pixels=47))

    assert exc_info.value.code == "source_total_pixel_limit_exceeded"


def test_unsupported_decodable_image_is_rejected(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "source.bmp", image_format="BMP")

    with pytest.raises(SourceNormalizationError) as exc_info:
        normalize_document_source(path, limits=_limits())

    assert exc_info.value.code == "source_media_type_unsupported"


def test_malformed_source_has_stable_failure(tmp_path: Path) -> None:
    path = tmp_path / "malformed.png"
    path.write_bytes(b"not an image")

    with pytest.raises(SourceNormalizationError) as exc_info:
        normalize_document_source(path, limits=_limits())

    assert exc_info.value.code == "source_malformed"


def test_malformed_pdf_has_stable_failure(tmp_path: Path) -> None:
    path = tmp_path / "malformed.pdf"
    path.write_bytes(b"%PDF-1.7\nnot a valid PDF")

    with pytest.raises(SourceNormalizationError) as exc_info:
        normalize_document_source(path, limits=_limits())

    assert exc_info.value.code == "source_malformed"


def test_empty_source_has_stable_failure(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")

    with pytest.raises(SourceNormalizationError) as exc_info:
        normalize_document_source(path, limits=_limits())

    assert exc_info.value.code == "source_malformed"


def test_normalized_source_rejects_unstable_page_indices(tmp_path: Path) -> None:
    source = normalize_document_source(
        _write_image(tmp_path / "source.png", image_format="PNG"),
        limits=_limits(),
    )

    with pytest.raises(ValueError, match="contiguous and ordered"):
        NormalizedDocumentSource(
            source_path=source.source_path,
            source_media_type=source.source_media_type,
            pages=(replace(source.pages[0], page_index=1),),
        )
