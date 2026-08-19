"""OpenAI implementation of the provider-neutral embedding gateway."""

from __future__ import annotations

import time
from collections.abc import Sequence
from threading import RLock
from typing import Any

import requests
from pydantic import ValidationError

from receipt_intelligence.application.ports.embeddings import (
    EmbeddingBatchResult,
    EmbeddingProviderError,
)
from receipt_intelligence.application.ports.llm import ModelCallMetrics

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIEmbeddingGateway:
    """HTTP adapter for OpenAI's ``POST /embeddings`` endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        dimensions: int | None = None,
        timeout_seconds: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        normalized_key = str(api_key or "").strip()
        normalized_model = str(model or "").strip()
        normalized_url = str(base_url or "").strip().rstrip("/")

        if not normalized_key:
            raise ValueError("api_key must not be empty.")
        if not normalized_model:
            raise ValueError("model must not be empty.")
        if not normalized_url:
            raise ValueError("base_url must not be empty.")
        if dimensions is not None and dimensions <= 0:
            raise ValueError("dimensions must be positive when provided.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        self.model = normalized_model
        self.base_url = normalized_url
        self.dimensions = int(dimensions) if dimensions is not None else None
        self.timeout_seconds = float(timeout_seconds)
        self._api_key = normalized_key
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._lock = RLock()
        self._closed = False

    def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        normalized_texts = self._normalize_texts(texts)
        if not normalized_texts:
            return EmbeddingBatchResult.empty(model=self.model)

        payload: dict[str, Any] = {
            "model": self.model,
            "input": normalized_texts,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        request_started = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("OpenAI embedding gateway is closed.")
            try:
                response = self._session.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise EmbeddingProviderError(f"OpenAI embedding request failed: {exc}") from exc

            try:
                response_payload = response.json()
            except (TypeError, ValueError) as exc:
                raise EmbeddingProviderError("OpenAI returned invalid JSON for /embeddings.") from exc

        if not isinstance(response_payload, dict):
            raise EmbeddingProviderError("OpenAI embedding response must be a JSON object.")

        raw_data = response_payload.get("data")
        if not isinstance(raw_data, list):
            raise EmbeddingProviderError("OpenAI response contains no embedding data array.")
        if len(raw_data) != len(normalized_texts):
            raise EmbeddingProviderError(
                "Embedding count does not match input count: "
                f"received {len(raw_data)}, expected {len(normalized_texts)}."
            )

        raw_vectors = self._vectors_in_input_order(raw_data, expected=len(normalized_texts))
        dimension = len(raw_vectors[0]) if raw_vectors else 0
        response_model = str(response_payload.get("model") or self.model)
        usage = response_payload.get("usage")
        prompt_tokens = (
            self._optional_nonnegative_int(usage.get("prompt_tokens"))
            if isinstance(usage, dict)
            else None
        )
        request_duration_ms = (time.perf_counter() - request_started) * 1000.0
        model_metrics = ModelCallMetrics(
            provider="openai",
            endpoint="embed",
            model=response_model,
            request_duration_ms=request_duration_ms,
            prompt_eval_count=prompt_tokens,
            input_count=len(normalized_texts),
        )

        try:
            return EmbeddingBatchResult.model_validate(
                {
                    "model": response_model,
                    "vectors": raw_vectors,
                    "dimension": dimension,
                    "prompt_eval_count": prompt_tokens,
                    "model_calls": [model_metrics],
                }
            )
        except ValidationError as exc:
            raise EmbeddingProviderError(f"OpenAI returned invalid embeddings: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_session:
                self._session.close()

    def __enter__(self) -> OpenAIEmbeddingGateway:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _vectors_in_input_order(raw_data: list[Any], *, expected: int) -> list[list[Any]]:
        ordered: list[list[Any] | None] = [None] * expected
        for entry in raw_data:
            if not isinstance(entry, dict):
                raise EmbeddingProviderError("Each OpenAI embedding entry must be an object.")
            index = entry.get("index")
            vector = entry.get("embedding")
            if isinstance(index, bool) or not isinstance(index, int):
                raise EmbeddingProviderError("Each OpenAI embedding entry requires an integer index.")
            if index < 0 or index >= expected:
                raise EmbeddingProviderError(f"OpenAI returned out-of-range embedding index {index}.")
            if ordered[index] is not None:
                raise EmbeddingProviderError(f"OpenAI returned duplicate embedding index {index}.")
            if not isinstance(vector, list):
                raise EmbeddingProviderError("Each OpenAI embedding must be a numeric array.")
            ordered[index] = vector

        missing = [index for index, vector in enumerate(ordered) if vector is None]
        if missing:
            raise EmbeddingProviderError(
                "OpenAI response is missing embedding indexes: "
                + ", ".join(str(index) for index in missing)
            )
        return [vector for vector in ordered if vector is not None]

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


__all__ = ["OpenAIEmbeddingGateway"]
