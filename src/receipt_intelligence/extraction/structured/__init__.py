"""Gemma scalar/item structured-extraction subsystem."""

from receipt_intelligence.extraction.structured.assembler import assemble_receipt
from receipt_intelligence.extraction.structured.composition import (
    build_gemma_structured_extraction_service,
)
from receipt_intelligence.extraction.structured.service import GemmaStructuredExtractionService

__all__ = [
    "GemmaStructuredExtractionService",
    "assemble_receipt",
    "build_gemma_structured_extraction_service",
]
