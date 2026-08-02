"""Typed contracts for Gemma scalar/item extraction and receipt assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from receipt_intelligence.application.ports.llm import ModelCallMetrics
from receipt_intelligence.extraction.contracts.common import JsonObject, StageArtifact
from receipt_intelligence.extraction.contracts.transcription import TranscriptionResult


class GemmaTaskStatus(StrEnum):
    COMPLETED = "completed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class StructuredExtractionRequest:
    run_id: str
    transcription: TranscriptionResult

    def __post_init__(self) -> None:
        run_id = str(self.run_id or "").strip()
        if not run_id:
            raise ValueError("StructuredExtractionRequest.run_id must not be empty.")
        object.__setattr__(self, "run_id", run_id)


@dataclass(frozen=True, slots=True)
class GemmaTaskResult:
    task_name: str
    prompt_id: str
    status: GemmaTaskStatus
    answer: JsonObject | None = None
    raw_model_content: str | None = None
    thinking: str | None = None
    metrics: ModelCallMetrics | None = None
    error: str | None = None
    diagnostics: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        task_name = str(self.task_name or "").strip()
        prompt_id = str(self.prompt_id or "").strip()
        if not task_name or not prompt_id:
            raise ValueError("Gemma task identifiers must not be empty.")
        if self.status is GemmaTaskStatus.COMPLETED and not isinstance(self.answer, dict):
            raise ValueError("Completed Gemma tasks require an object answer.")
        object.__setattr__(self, "task_name", task_name)
        object.__setattr__(self, "prompt_id", prompt_id)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True, slots=True)
class StructuredExtractionResult:
    receipt: JsonObject
    scalar_results: tuple[GemmaTaskResult, ...] = ()
    item_result: GemmaTaskResult | None = None
    item_contract: JsonObject = field(default_factory=dict)
    missing_scalar_tasks: tuple[str, ...] = ()
    diagnostics: JsonObject = field(default_factory=dict)
    artifacts: tuple[StageArtifact, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, dict):
            raise TypeError("StructuredExtractionResult.receipt must be a dictionary.")
        object.__setattr__(self, "receipt", dict(self.receipt))
        object.__setattr__(self, "item_contract", dict(self.item_contract))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def model_calls(self) -> tuple[ModelCallMetrics, ...]:
        calls = [result.metrics for result in self.scalar_results if result.metrics is not None]
        if self.item_result is not None and self.item_result.metrics is not None:
            calls.append(self.item_result.metrics)
        return tuple(calls)


__all__ = [
    "GemmaTaskResult",
    "GemmaTaskStatus",
    "StructuredExtractionRequest",
    "StructuredExtractionResult",
]
