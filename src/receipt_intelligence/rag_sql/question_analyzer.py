"""LLM question analysis for the isolated RAG-SQL strategy."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from receipt_intelligence.extraction.parsing.llm_parser import (
    ollama_generate,
    parse_json_from_llm,
)
from receipt_intelligence.observability.ollama import (
    OllamaCallMetrics,
    get_ollama_metrics,
)
from receipt_intelligence.prompts import render_prompt_template
from receipt_intelligence.rag_sql.models import QuestionAnalysisPayload, QuestionAnalysisResult

GenerateFunction = Callable[..., str]


class QuestionAnalysisError(RuntimeError):
    """Raised when no valid structured analysis is produced."""

    def __init__(
        self,
        message: str,
        *,
        ollama_calls: list[OllamaCallMetrics] | None = None,
    ) -> None:
        super().__init__(message)
        self.ollama_calls = list(ollama_calls or [])


@dataclass(frozen=True)
class QuestionAnalyzerConfig:
    enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    model: str = "gemma4"
    num_ctx: int = 3072
    num_predict: int = 768
    timeout_seconds: float = 120.0
    retry_count: int = 1
    format_json: bool = True
    keep_alive: str | None = None
    maximum_entities: int = 4

    def __post_init__(self) -> None:
        if not str(self.ollama_url or "").strip():
            raise ValueError("ollama_url must not be empty.")
        if not str(self.model or "").strip():
            raise ValueError("model must not be empty.")
        if self.num_ctx <= 0 or self.num_predict <= 0:
            raise ValueError("num_ctx and num_predict must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.retry_count < 0:
            raise ValueError("retry_count must not be negative.")
        if self.maximum_entities <= 0 or self.maximum_entities > 4:
            raise ValueError("maximum_entities must be between 1 and 4.")


class RagSqlQuestionAnalyzer:
    def __init__(
        self,
        config: QuestionAnalyzerConfig,
        *,
        generate: GenerateFunction = ollama_generate,
    ) -> None:
        self.config = config
        self.generate = generate

    def analyze(self, question: str) -> QuestionAnalysisResult:
        normalized_question = " ".join(str(question or "").split()).strip()
        if not normalized_question:
            raise ValueError("question must not be empty.")
        if not self.config.enabled:
            raise QuestionAnalysisError("RAG-SQL question analysis is disabled.")

        started = time.perf_counter()
        previous_error: str | None = None
        last_error: Exception | None = None
        attempts = max(1, self.config.retry_count + 1)
        ollama_calls: list[OllamaCallMetrics] = []

        for attempt in range(1, attempts + 1):
            retry_block = ""
            if previous_error:
                retry_block = (
                    "Previous response validation error:\n"
                    f"{previous_error}\n"
                    "Correct the JSON response without changing the user question."
                )
            prompt = render_prompt_template(
                "rag_sql_question_analyzer.txt",
                QUESTION=normalized_question,
                RETRY_BLOCK=retry_block,
            )
            try:
                raw = self.generate(
                    ollama_url=self.config.ollama_url,
                    model=self.config.model,
                    prompt=prompt,
                    num_ctx=self.config.num_ctx,
                    num_predict=self.config.num_predict,
                    temperature=0.0,
                    keep_alive=self.config.keep_alive,
                    timeout=self.config.timeout_seconds,
                    format_json=self.config.format_json,
                )
                call_metrics = get_ollama_metrics(raw)
                if call_metrics is not None:
                    ollama_calls.append(call_metrics)
                payload = QuestionAnalysisPayload.model_validate(parse_json_from_llm(raw))
                if len(payload.entities) > self.config.maximum_entities:
                    raise ValueError(
                        f"The analysis returned {len(payload.entities)} entities; maximum is "
                        f"{self.config.maximum_entities}."
                    )
                return QuestionAnalysisResult(
                    **payload.model_dump(mode="python"),
                    model=self.config.model,
                    attempts=attempt,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    ollama_calls=ollama_calls,
                )
            except Exception as exc:
                last_error = exc
                previous_error = f"{type(exc).__name__}: {exc}"

        duration_ms = (time.perf_counter() - started) * 1000.0
        raise QuestionAnalysisError(
            "RAG-SQL question analysis failed after "
            f"{attempts} attempt(s) in {duration_ms:.1f} ms: "
            f"{type(last_error).__name__ if last_error else 'UnknownError'}: {last_error}",
            ollama_calls=ollama_calls,
        ) from last_error


__all__ = [
    "QuestionAnalysisError",
    "QuestionAnalyzerConfig",
    "RagSqlQuestionAnalyzer",
]
