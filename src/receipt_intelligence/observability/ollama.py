"""Typed Ollama timing metrics used by RAG-SQL diagnostics.

Ollama returns nanosecond timing fields for completed generation and embedding
requests.  The helpers in this module preserve those raw values and expose a
compact millisecond representation for application diagnostics.  They do not
log prompts, generated text, embeddings, or receipt content.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OllamaCallMetrics(BaseModel):
    """Provider and client timing information for one Ollama request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

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
    def validate_generation_fields(self) -> Self:
        if self.endpoint == "embed" and self.eval_count not in (None, 0):
            raise ValueError("Embedding metrics cannot contain generated-token counts.")
        return self

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        endpoint: Literal["generate", "embed"],
        model: str,
        request_duration_ms: float,
        input_count: int | None = None,
    ) -> OllamaCallMetrics:
        """Build metrics from an Ollama response while tolerating absent fields."""

        return cls(
            endpoint=endpoint,
            model=str(payload.get("model") or model),
            request_duration_ms=max(0.0, float(request_duration_ms)),
            total_duration_ns=_optional_nonnegative_int(payload.get("total_duration")),
            load_duration_ns=_optional_nonnegative_int(payload.get("load_duration")),
            prompt_eval_count=_optional_nonnegative_int(payload.get("prompt_eval_count")),
            prompt_eval_duration_ns=_optional_nonnegative_int(payload.get("prompt_eval_duration")),
            eval_count=(
                _optional_nonnegative_int(payload.get("eval_count"))
                if endpoint == "generate"
                else None
            ),
            eval_duration_ns=(
                _optional_nonnegative_int(payload.get("eval_duration"))
                if endpoint == "generate"
                else None
            ),
            done_reason=(
                str(payload.get("done_reason") or "").strip() or None
                if endpoint == "generate"
                else None
            ),
            input_count=input_count,
        )

    def to_diagnostics(self) -> dict[str, Any]:
        """Return raw counters plus readable durations and throughput."""

        diagnostics: dict[str, Any] = {
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


class OllamaTextResponse(str):
    """String-compatible generation response carrying provider metrics."""

    ollama_metrics: OllamaCallMetrics

    def __new__(
        cls,
        value: str,
        *,
        metrics: OllamaCallMetrics,
    ) -> OllamaTextResponse:
        instance = str.__new__(cls, value)
        instance.ollama_metrics = metrics
        return instance


def get_ollama_metrics(value: object) -> OllamaCallMetrics | None:
    """Return metrics attached to an Ollama text response, if present."""

    metrics = getattr(value, "ollama_metrics", None)
    return metrics if isinstance(metrics, OllamaCallMetrics) else None


def metrics_to_diagnostics(
    calls: list[OllamaCallMetrics] | tuple[OllamaCallMetrics, ...],
) -> list[dict[str, Any]]:
    """Serialize multiple calls for one application stage."""

    return [call.to_diagnostics() for call in calls]


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _ns_to_ms(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / 1_000_000.0, 3)


def _tokens_per_second(count: int | None, duration_ns: int | None) -> float | None:
    if count is None or duration_ns is None or duration_ns <= 0:
        return None
    return round(float(count) / (duration_ns / 1_000_000_000.0), 2)


__all__ = [
    "OllamaCallMetrics",
    "OllamaTextResponse",
    "get_ollama_metrics",
    "metrics_to_diagnostics",
]
