"""Provider-neutral language-model contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelCallMetrics(BaseModel):
    """Timing and token counters for one model-provider request.

    Provider-specific adapters may populate the optional raw timing fields, but
    feature packages depend only on this neutral contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    provider: str = Field(default="unknown", min_length=1, max_length=100)
    endpoint: Literal["generate", "embed"]
    model: str = Field(min_length=1, max_length=200)
    request_duration_ms: float = Field(default=0.0, ge=0.0)
    total_duration_ns: int | None = Field(default=None, ge=0)
    load_duration_ns: int | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    prompt_eval_duration_ns: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    eval_duration_ns: int | None = Field(default=None, ge=0)
    done_reason: str | None = Field(default=None, max_length=100)
    input_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_generation_fields(self) -> ModelCallMetrics:
        if self.endpoint == "embed" and self.eval_count not in (None, 0):
            raise ValueError("Embedding metrics cannot contain generated-token counts.")
        return self

    def to_diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "request_duration_ms": round(self.request_duration_ms, 3),
            "total_duration_ms": _ns_to_ms(self.total_duration_ns),
            "load_duration_ms": _ns_to_ms(self.load_duration_ns),
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration_ms": _ns_to_ms(self.prompt_eval_duration_ns),
            "eval_count": self.eval_count,
            "eval_duration_ms": _ns_to_ms(self.eval_duration_ns),
            "done_reason": self.done_reason,
            "input_count": self.input_count,
        }
        prompt_rate = _tokens_per_second(
            self.prompt_eval_count,
            self.prompt_eval_duration_ns,
        )
        eval_rate = _tokens_per_second(self.eval_count, self.eval_duration_ns)
        if prompt_rate is not None:
            diagnostics["prompt_tokens_per_second"] = prompt_rate
        if eval_rate is not None:
            diagnostics["generated_tokens_per_second"] = eval_rate
        return {key: value for key, value in diagnostics.items() if value is not None}


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    model: str
    prompt: str
    num_ctx: int = 24384
    num_predict: int = 8192
    temperature: float = 0.0
    keep_alive: str | None = None
    timeout_seconds: float = 240.0
    format_json: bool = True


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    metrics: ModelCallMetrics | None = None

    def __post_init__(self) -> None:
        normalized = str(self.text or "").strip()
        if not normalized:
            raise ValueError("GenerationResult.text must not be empty.")
        object.__setattr__(self, "text", normalized)


GenerationValue: TypeAlias = GenerationResult | str


class LlmGateway(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...


def coerce_generation_result(value: GenerationValue) -> GenerationResult:
    """Normalize legacy string generators at an explicit compatibility edge."""

    if isinstance(value, GenerationResult):
        return value
    if isinstance(value, str):
        return GenerationResult(text=value)
    raise TypeError(
        "LLM generator must return GenerationResult or str, "
        f"got {type(value).__name__}."
    )


def metrics_to_diagnostics(
    calls: list[ModelCallMetrics] | tuple[ModelCallMetrics, ...],
) -> list[dict[str, Any]]:
    return [call.to_diagnostics() for call in calls]


def _ns_to_ms(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / 1_000_000.0, 3)


def _tokens_per_second(count: int | None, duration_ns: int | None) -> float | None:
    if count is None or duration_ns is None or duration_ns <= 0:
        return None
    return round(float(count) / (duration_ns / 1_000_000_000.0), 2)


__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "GenerationValue",
    "LlmGateway",
    "ModelCallMetrics",
    "coerce_generation_result",
    "metrics_to_diagnostics",
]
