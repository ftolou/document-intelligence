from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

import receipt_intelligence.pipeline.integrated_receipt_pipeline as pipeline
from receipt_intelligence.extraction import (
    SourceImageValidationError,
    source_normalization,
    validate_source_image,
)
from receipt_intelligence.extraction.config import ExtractionRequest


def _write_png(path: Path, *, size: tuple[int, int] = (8, 6)) -> Path:
    Image.new("RGB", size, "white").save(path, format="PNG")
    return path


def _request_values(tmp_path: Path, source_image_path: Path) -> dict[str, object]:
    return {
        "source_image_path": source_image_path,
        "result_dir": tmp_path / "results",
        "run_id": "source-validation",
        "ollama_url": "http://ollama:11434",
        "model": "gemma",
    }


def test_valid_source_image_passes_bounded_full_decode(tmp_path: Path) -> None:
    path = _write_png(tmp_path / "receipt.png")

    validate_source_image(path, max_width=8, max_height=6, max_pixels=48)


@pytest.mark.parametrize(
    ("limits", "expected_code"),
    [
        (
            {"max_width": 7, "max_height": 6, "max_pixels": 48},
            "source_image_dimension_limit_exceeded",
        ),
        (
            {"max_width": 8, "max_height": 5, "max_pixels": 48},
            "source_image_dimension_limit_exceeded",
        ),
        (
            {"max_width": 8, "max_height": 6, "max_pixels": 47},
            "source_image_pixel_limit_exceeded",
        ),
    ],
)
def test_source_image_limits_reject_pathological_dimensions(
    tmp_path: Path,
    limits: dict[str, int],
    expected_code: str,
) -> None:
    path = _write_png(tmp_path / "receipt.png")

    with pytest.raises(SourceImageValidationError) as exc_info:
        validate_source_image(path, **limits)

    assert exc_info.value.code == expected_code


def test_exif_orientation_controls_effective_dimension_limits(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (8, 6), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, format="JPEG", exif=exif)

    validate_source_image(path, max_width=6, max_height=8, max_pixels=48)

    with pytest.raises(SourceImageValidationError) as exc_info:
        validate_source_image(path, max_width=8, max_height=6, max_pixels=48)

    assert exc_info.value.code == "source_image_dimension_limit_exceeded"


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("malformed.bin", b"not an image"),
        ("empty.bin", b""),
    ],
)
def test_malformed_source_bytes_use_stable_generic_failure(
    tmp_path: Path,
    filename: str,
    payload: bytes,
) -> None:
    path = tmp_path / filename
    path.write_bytes(payload)

    with pytest.raises(SourceImageValidationError) as exc_info:
        validate_source_image(path, max_width=100, max_height=100, max_pixels=10_000)

    assert exc_info.value.code == "source_image_malformed"


def test_truncated_source_image_is_rejected_after_decoder_validation(tmp_path: Path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="JPEG")
    data = buffer.getvalue()
    path = tmp_path / "truncated.jpg"
    path.write_bytes(data[: len(data) // 2])

    with pytest.raises(SourceImageValidationError) as exc_info:
        validate_source_image(path, max_width=100, max_height=100, max_pixels=10_000)

    assert exc_info.value.code == "source_image_malformed"


def test_missing_source_image_has_stable_not_found_failure(tmp_path: Path) -> None:
    with pytest.raises(SourceImageValidationError) as exc_info:
        validate_source_image(
            tmp_path / "missing.png",
            max_width=100,
            max_height=100,
            max_pixels=10_000,
        )

    assert exc_info.value.code == "source_image_not_found"


@pytest.mark.parametrize(
    "field_name",
    ["source_image_max_width", "source_image_max_height", "source_image_max_pixels"],
)
def test_extraction_request_requires_positive_source_image_limits(
    tmp_path: Path,
    field_name: str,
) -> None:
    values = _request_values(tmp_path, tmp_path / "receipt.png")
    values[field_name] = 0

    with pytest.raises(ValueError, match=field_name):
        ExtractionRequest(**values)


def test_extraction_request_has_conservative_default_source_limits(tmp_path: Path) -> None:
    request = ExtractionRequest(**_request_values(tmp_path, tmp_path / "receipt.png"))

    assert request.source_image_max_width == 12000
    assert request.source_image_max_height == 12000
    assert request.source_image_max_pixels == 40_000_000


def test_canonical_pipeline_validates_before_dependency_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_png(tmp_path / "receipt.png")
    request = ExtractionRequest(
        **_request_values(tmp_path, path),
        source_image_max_pixels=47,
    )

    def fail_generic_normalization(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Receipt extraction must not use generic source normalization.")

    def fail_dependency_construction(_request: ExtractionRequest) -> None:
        pytest.fail("Dependencies must not be constructed for an invalid source image.")

    monkeypatch.setattr(
        source_normalization,
        "normalize_document_source",
        fail_generic_normalization,
    )
    monkeypatch.setattr(
        pipeline,
        "normalize_document_source",
        fail_generic_normalization,
        raising=False,
    )
    monkeypatch.setattr(pipeline, "build_extraction_dependencies", fail_dependency_construction)

    with pytest.raises(SourceImageValidationError) as exc_info:
        pipeline.run_receipt_extraction(request)

    assert exc_info.value.code == "source_image_pixel_limit_exceeded"
