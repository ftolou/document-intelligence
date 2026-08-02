"""Composition helper for the inactive Gemma structured-extraction subsystem."""

from __future__ import annotations

from pathlib import Path

from receipt_intelligence.adapters.chat import OllamaChatGateway
from receipt_intelligence.extraction.settings import ParsingSettings, PipelineSettings
from receipt_intelligence.extraction.structured.service import GemmaStructuredExtractionService
from receipt_intelligence.extraction.structured.task_runner import GemmaTaskRunner
from receipt_intelligence.prompts.registry import default_prompt_registry


def build_gemma_structured_extraction_service(
    settings: PipelineSettings | ParsingSettings,
    *,
    result_dir: Path | None = None,
) -> GemmaStructuredExtractionService:
    if isinstance(settings, PipelineSettings):
        parsing = settings.parsing
        output_dir = settings.runtime.result_dir
    else:
        parsing = settings
        if result_dir is None:
            raise ValueError("result_dir is required with ParsingSettings.")
        output_dir = Path(result_dir)
    prompts = default_prompt_registry()
    return GemmaStructuredExtractionService(
        task_runner=GemmaTaskRunner(
            gateway=OllamaChatGateway(parsing.ollama_url),
            prompts=prompts,
            settings=parsing,
        ),
        settings=parsing,
        result_dir=output_dir,
    )


__all__ = ["build_gemma_structured_extraction_service"]
