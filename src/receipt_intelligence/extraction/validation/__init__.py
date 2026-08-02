"""Read-only deterministic validation engine for the next extraction pipeline."""

from receipt_intelligence.extraction.validation.composition import (
    build_deterministic_validation_service,
)
from receipt_intelligence.extraction.validation.engine import DeterministicValidationEngine

__all__ = ["DeterministicValidationEngine", "build_deterministic_validation_service"]
