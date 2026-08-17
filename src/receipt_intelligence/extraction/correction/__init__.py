"""Validator-gated specialist correction for receipt extraction."""

from receipt_intelligence.extraction.correction.composition import (
    build_specialist_correction_service,
)
from receipt_intelligence.extraction.correction.service import SpecialistCorrectionService

__all__ = ["SpecialistCorrectionService", "build_specialist_correction_service"]
