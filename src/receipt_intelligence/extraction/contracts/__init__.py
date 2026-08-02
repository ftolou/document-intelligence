"""Typed contracts exchanged by the next receipt-extraction stages."""

from receipt_intelligence.extraction.contracts.common import (
    JsonObject,
    ReadonlyJsonObject,
    StageArtifact,
)
from receipt_intelligence.extraction.contracts.correction import (
    CorrectionAttempt,
    CorrectionAttemptStatus,
    CorrectionRequest,
    CorrectionResult,
    CorrectionTargetOutcome,
)
from receipt_intelligence.extraction.contracts.extraction import (
    GemmaTaskResult,
    GemmaTaskStatus,
    StructuredExtractionRequest,
    StructuredExtractionResult,
)
from receipt_intelligence.extraction.contracts.presentation import (
    CategorizationRequest,
    CategorizationResult,
    CategorizationStatus,
    FinalizationRequest,
    FinalizationResult,
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
    "CategorizationRequest",
    "CategorizationResult",
    "CategorizationStatus",
    "CorrectionAttempt",
    "CorrectionAttemptStatus",
    "CorrectionRequest",
    "CorrectionResult",
    "CorrectionTargetOutcome",
    "FinalizationRequest",
    "FinalizationResult",
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
