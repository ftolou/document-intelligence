"""Typed contracts exchanged by the next receipt-extraction stages."""

from receipt_intelligence.extraction.contracts.common import (
    JsonObject,
    ReadonlyJsonObject,
    StageArtifact,
)
from receipt_intelligence.extraction.contracts.correction import (
    CorrectionAttempt,
    CorrectionAttemptStatus,
    CorrectionResult,
    CorrectionTargetOutcome,
)
from receipt_intelligence.extraction.contracts.extraction import (
    GemmaTaskResult,
    GemmaTaskStatus,
    StructuredExtractionRequest,
    StructuredExtractionResult,
)
from receipt_intelligence.extraction.contracts.result import ReceiptPipelineResult
from receipt_intelligence.extraction.contracts.transcription import (
    BoundingBox,
    CanonicalTranscriptionRow,
    ReceiptCrop,
    TranscriptionFragment,
    TranscriptionRequest,
    TranscriptionResult,
)
from receipt_intelligence.extraction.contracts.validation import (
    ValidationCheck,
    ValidationReport,
    ValidationRequest,
    ValidationSeverity,
    ValidationStatus,
)

__all__ = [
    "BoundingBox",
    "CanonicalTranscriptionRow",
    "CorrectionAttempt",
    "CorrectionAttemptStatus",
    "CorrectionResult",
    "CorrectionTargetOutcome",
    "GemmaTaskResult",
    "GemmaTaskStatus",
    "JsonObject",
    "ReadonlyJsonObject",
    "ReceiptCrop",
    "ReceiptPipelineResult",
    "StageArtifact",
    "StructuredExtractionRequest",
    "StructuredExtractionResult",
    "TranscriptionFragment",
    "TranscriptionRequest",
    "TranscriptionResult",
    "ValidationCheck",
    "ValidationReport",
    "ValidationRequest",
    "ValidationSeverity",
    "ValidationStatus",
]
