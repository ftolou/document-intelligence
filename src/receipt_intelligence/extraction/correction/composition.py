"""Composition helpers for the specialist correction subsystem."""

from __future__ import annotations

from pathlib import Path

from receipt_intelligence.application.ports.chat import ChatGateway
from receipt_intelligence.extraction.correction.artifacts import (
    CorrectionArtifactSink,
    NullCorrectionArtifactSink,
)
from receipt_intelligence.extraction.correction.invocation import PromptBoundSourceEvidenceInvoker
from receipt_intelligence.extraction.correction.profile import load_correction_profile
from receipt_intelligence.extraction.correction.service import SpecialistCorrectionService
from receipt_intelligence.extraction.services.validation import ReceiptValidationService
from receipt_intelligence.extraction.settings import CorrectionSettings
from receipt_intelligence.prompts.registry import PromptRegistry


def build_specialist_correction_service(
    *,
    gateway: ChatGateway,
    prompts: PromptRegistry,
    validation_service: ReceiptValidationService,
    settings: CorrectionSettings,
    artifact_sink: CorrectionArtifactSink | None = None,
) -> SpecialistCorrectionService:
    profile_path = settings.profile_path or (
        Path(__file__).resolve().parent / "config" / "production.json"
    )
    profile = load_correction_profile(profile_path)
    return SpecialistCorrectionService(
        profile=profile,
        invoker=PromptBoundSourceEvidenceInvoker(
            gateway=gateway,
            prompts=prompts,
            settings=settings,
        ),
        validation_service=validation_service,
        artifact_sink=artifact_sink or NullCorrectionArtifactSink(),
        enabled=settings.enabled,
    )


__all__ = ["build_specialist_correction_service"]
