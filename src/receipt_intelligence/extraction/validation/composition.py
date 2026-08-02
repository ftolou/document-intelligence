"""Composition helper for the pure deterministic validation service."""

from receipt_intelligence.extraction.settings import ValidationSettings
from receipt_intelligence.extraction.validation.engine import DeterministicValidationEngine


def build_deterministic_validation_service(
    settings: ValidationSettings | None = None,
) -> DeterministicValidationEngine:
    # Settings are supplied to each ValidationRequest so the engine remains stateless.
    del settings
    return DeterministicValidationEngine()


__all__ = ["build_deterministic_validation_service"]
