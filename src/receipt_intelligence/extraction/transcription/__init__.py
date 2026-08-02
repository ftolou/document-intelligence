"""Paddle-geometry/Qwen transcription implementation for the next pipeline."""

from receipt_intelligence.extraction.transcription.canonical import (
    build_canonical_rows,
    clean_plain_lines,
    serialize_canonical_rows,
)
from receipt_intelligence.extraction.transcription.composition import (
    build_canonical_transcription_service,
)
from receipt_intelligence.extraction.transcription.crop_planner import (
    determine_effective_crop_count,
    full_image_crop,
    plan_safe_crops,
)
from receipt_intelligence.extraction.transcription.line_clustering import cluster_text_regions
from receipt_intelligence.extraction.transcription.service import (
    CanonicalReceiptTranscriptionService,
)

__all__ = [
    "CanonicalReceiptTranscriptionService",
    "build_canonical_rows",
    "build_canonical_transcription_service",
    "clean_plain_lines",
    "cluster_text_regions",
    "determine_effective_crop_count",
    "full_image_crop",
    "plan_safe_crops",
    "serialize_canonical_rows",
]
