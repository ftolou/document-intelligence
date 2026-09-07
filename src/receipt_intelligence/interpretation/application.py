"""Public application entry point for generic document interpretation."""

from __future__ import annotations

from pathlib import Path

from receipt_intelligence.application.ports.multimodal import MultimodalGateway
from receipt_intelligence.extraction.source_normalization import SourceNormalizationLimits
from receipt_intelligence.interpretation.contracts import (
    DocumentInterpretationOutcome,
    DocumentInterpretationRequest,
)
from receipt_intelligence.interpretation.workflow import (
    DEFAULT_INTERPRETATION_MAX_OUTPUT_TOKENS,
    OnePassDocumentInterpreter,
)


def run_document_interpretation(
    request: DocumentInterpretationRequest,
    source_path: str | Path,
    *,
    gateway: MultimodalGateway,
    model: str,
    source_limits: SourceNormalizationLimits,
    max_output_tokens: int = DEFAULT_INTERPRETATION_MAX_OUTPUT_TOKENS,
) -> DocumentInterpretationOutcome:
    """Interpret one document through the provider-neutral Core workflow.

    Runtime composition supplies the model gateway, opaque model identifier, source
    bounds, and maximum output-token budget. Provider-neutral generation and
    source-normalization failures propagate unchanged to the caller.
    """

    return OnePassDocumentInterpreter(
        gateway=gateway,
        model=model,
        source_limits=source_limits,
        max_output_tokens=max_output_tokens,
    ).interpret(request, source_path)


__all__ = ["run_document_interpretation"]
