"""Typed contracts for Paddle detection, crop planning, and Qwen transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from receipt_intelligence.application.ports.llm import ModelCallMetrics
from receipt_intelligence.extraction.contracts.common import JsonObject, StageArtifact

BoundingBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    source_image_path: Path
    run_id: str
    legacy_ocr_json_path: Path | None = None

    def __post_init__(self) -> None:
        run_id = str(self.run_id or "").strip()
        if not run_id:
            raise ValueError("TranscriptionRequest.run_id must not be empty.")
        object.__setattr__(self, "source_image_path", Path(self.source_image_path))
        object.__setattr__(self, "run_id", run_id)
        if self.legacy_ocr_json_path is not None:
            object.__setattr__(self, "legacy_ocr_json_path", Path(self.legacy_ocr_json_path))


@dataclass(frozen=True, slots=True)
class ReceiptCrop:
    crop_id: str
    image_path: Path
    source_box: BoundingBox
    order: int
    is_full_image_fallback: bool = False
    detected_line_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        crop_id = str(self.crop_id or "").strip()
        if not crop_id:
            raise ValueError("ReceiptCrop.crop_id must not be empty.")
        if self.order < 0:
            raise ValueError("ReceiptCrop.order must not be negative.")
        object.__setattr__(self, "crop_id", crop_id)
        object.__setattr__(self, "image_path", Path(self.image_path))


@dataclass(frozen=True, slots=True)
class TranscriptionFragment:
    crop_id: str
    text: str
    order: int
    metrics: ModelCallMetrics | None = None
    attempt: int = 1
    text_source: str | None = None

    def __post_init__(self) -> None:
        crop_id = str(self.crop_id or "").strip()
        text = str(self.text or "").strip()
        if not crop_id:
            raise ValueError("TranscriptionFragment.crop_id must not be empty.")
        if not text:
            raise ValueError("TranscriptionFragment.text must not be empty.")
        if self.order < 0:
            raise ValueError("TranscriptionFragment.order must not be negative.")
        if self.attempt < 1:
            raise ValueError("TranscriptionFragment.attempt must be positive.")
        object.__setattr__(self, "crop_id", crop_id)
        object.__setattr__(self, "text", text)
        if self.text_source is not None:
            object.__setattr__(self, "text_source", str(self.text_source).strip() or None)


@dataclass(frozen=True, slots=True)
class CanonicalTranscriptionRow:
    row_id: str
    text: str
    source_crop_ids: tuple[str, ...] = ()
    source_box: BoundingBox | None = None

    def __post_init__(self) -> None:
        row_id = str(self.row_id or "").strip()
        text = str(self.text or "").strip()
        if not row_id:
            raise ValueError("CanonicalTranscriptionRow.row_id must not be empty.")
        if not text:
            raise ValueError("CanonicalTranscriptionRow.text must not be empty.")
        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "text", text)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    canonical_text: str
    rows: tuple[CanonicalTranscriptionRow, ...]
    crops: tuple[ReceiptCrop, ...]
    fragments: tuple[TranscriptionFragment, ...]
    diagnostics: JsonObject = field(default_factory=dict)
    artifacts: tuple[StageArtifact, ...] = ()

    def __post_init__(self) -> None:
        canonical_text = str(self.canonical_text or "").strip()
        if not canonical_text:
            raise ValueError("TranscriptionResult.canonical_text must not be empty.")
        object.__setattr__(self, "canonical_text", canonical_text)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def used_full_image_fallback(self) -> bool:
        """Whether transcription used a whole-image fallback crop."""
        return any(crop.is_full_image_fallback for crop in self.crops)

__all__ = [
    "BoundingBox",
    "CanonicalTranscriptionRow",
    "ReceiptCrop",
    "TranscriptionFragment",
    "TranscriptionRequest",
    "TranscriptionResult",
]
