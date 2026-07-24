"""Ollama client for deterministic batch embedding generation."""

from __future__ import annotations

import time
from collections.abc import Sequence
from threading import RLock
from typing import Any

import requests
from pydantic import ValidationError

from receipt_intelligence.adapters.llm import model_metrics_from_ollama_payload
from receipt_intelligence.rag.models import EmbeddingBatchResult


class EmbeddingClientError(RuntimeError):
    """Raised when the embedding provider returns an unusable response."""


class OllamaEmbeddingClient:
    """Small, testable client for Ollama's ``POST /api/embed`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        keep_alive: str | None = "10m",
        session: requests.Session | None = None,
    ) -> None:
        normalized_url = str(base_url or "").strip().rstrip("/")
        normalized_model = str(model or "").strip()

        if not normalized_url:
            raise ValueError("base_url must not be empty.")
        if not normalized_model:
            raise ValueError("model must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        self.base_url = normalized_url
        self.model = normalized_model
        self.timeout_seconds = float(timeout_seconds)
        self.keep_alive = str(keep_alive).strip() if keep_alive is not None else None
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._lock = RLock()
        self._closed = False

    def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        """Embed a batch of non-empty strings and validate the provider output."""

        normalized_texts = self._normalize_texts(texts)
        if not normalized_texts:
            return EmbeddingBatchResult.empty(model=self.model)

        payload: dict[str, Any] = {
            "model": self.model,
            "input": normalized_texts,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        request_started = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("Ollama embedding client is closed.")
            try:
                response = self._session.post(
                    f"{self.base_url}/api/embed",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise EmbeddingClientError(f"Ollama embedding request failed: {exc}") from exc

            try:
                response_payload = response.json()
            except (TypeError, ValueError) as exc:
                raise EmbeddingClientError("Ollama returned invalid JSON for /api/embed.") from exc

        if not isinstance(response_payload, dict):
            raise EmbeddingClientError("Ollama embedding response must be a JSON object.")

        request_duration_ms = (time.perf_counter() - request_started) * 1000.0
        model_metrics = model_metrics_from_ollama_payload(
            response_payload,
            endpoint="embed",
            model=self.model,
            request_duration_ms=request_duration_ms,
            input_count=len(normalized_texts),
        )

        raw_vectors = response_payload.get("embeddings")
        if not isinstance(raw_vectors, list):
            raise EmbeddingClientError("Ollama response contains no embeddings array.")
        if len(raw_vectors) != len(normalized_texts):
            raise EmbeddingClientError(
                "Embedding count does not match input count: "
                f"received {len(raw_vectors)}, expected {len(normalized_texts)}."
            )

        dimension = 0
        if raw_vectors:
            first_vector = raw_vectors[0]
            if not isinstance(first_vector, list):
                raise EmbeddingClientError("Each embedding must be a numeric array.")
            dimension = len(first_vector)

        try:
            return EmbeddingBatchResult.model_validate(
                {
                    "model": str(response_payload.get("model") or self.model),
                    "vectors": raw_vectors,
                    "dimension": dimension,
                    "total_duration_ns": self._optional_nonnegative_int(
                        response_payload.get("total_duration")
                    ),
                    "load_duration_ns": self._optional_nonnegative_int(
                        response_payload.get("load_duration")
                    ),
                    "prompt_eval_count": self._optional_nonnegative_int(
                        response_payload.get("prompt_eval_count")
                    ),
                    "prompt_eval_duration_ns": self._optional_nonnegative_int(
                        response_payload.get("prompt_eval_duration")
                    ),
                    "ollama_calls": [model_metrics],
                }
            )
        except ValidationError as exc:
            raise EmbeddingClientError(f"Ollama returned invalid embeddings: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_session:
                self._session.close()

    def __enter__(self) -> OllamaEmbeddingClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _normalize_texts(texts: Sequence[str]) -> list[str]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string.")

        normalized: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"Embedding input {index} is not a string.")
            value = text.strip()
            if not value:
                raise ValueError(f"Embedding input {index} is empty.")
            normalized.append(value)
        return normalized

    @staticmethod
    def _optional_nonnegative_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None
