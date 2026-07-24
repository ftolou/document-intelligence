"""Shared invocation helper for gateway and legacy callable compatibility."""

from __future__ import annotations

from collections.abc import Callable

from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    GenerationResult,
    GenerationValue,
    LlmGateway,
    coerce_generation_result,
)

LegacyGenerateFunction = Callable[..., GenerationValue]


def invoke_generation(
    *,
    request: GenerationRequest,
    gateway: LlmGateway | None,
    legacy_generate: LegacyGenerateFunction | None,
    legacy_base_url: str,
) -> GenerationResult:
    if gateway is not None:
        return gateway.generate(request)
    if legacy_generate is None:
        raise RuntimeError("No LLM gateway was configured for this operation.")
    return coerce_generation_result(
        legacy_generate(
            ollama_url=legacy_base_url,
            model=request.model,
            prompt=request.prompt,
            num_ctx=request.num_ctx,
            num_predict=request.num_predict,
            temperature=request.temperature,
            keep_alive=request.keep_alive,
            timeout=request.timeout_seconds,
            format_json=request.format_json,
        )
    )


__all__ = ["LegacyGenerateFunction", "invoke_generation"]
