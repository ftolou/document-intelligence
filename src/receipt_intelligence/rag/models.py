"""Typed models for the receipt semantic-embedding foundation.

The embedding layer is a rebuildable semantic index. Financial values, dates,
and receipt totals remain structured data in SQLite and are intentionally not
part of these models.
"""

from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from receipt_intelligence.observability.ollama import OllamaCallMetrics


class StrictModel(BaseModel):
    """Base model that rejects unexpected data at integration boundaries."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EmbeddingBatchResult(StrictModel):
    """Validated batch returned by an embedding provider.

    All vectors in one batch must have the same dimension and contain only
    finite numbers. Empty batches are represented with ``dimension == 0``.
    """

    model: str = Field(min_length=1, max_length=200)
    vectors: list[list[float]] = Field(default_factory=list)
    dimension: int = Field(ge=0)
    total_duration_ns: int | None = Field(default=None, ge=0)
    load_duration_ns: int | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    prompt_eval_duration_ns: int | None = Field(default=None, ge=0)
    ollama_calls: list[OllamaCallMetrics] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_vectors(self) -> Self:
        if not self.vectors:
            if self.dimension != 0:
                raise ValueError("An empty embedding batch must have dimension=0.")
            return self

        if self.dimension <= 0:
            raise ValueError("A non-empty embedding batch requires a positive dimension.")

        for index, vector in enumerate(self.vectors):
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Embedding {index} has dimension {len(vector)}, expected {self.dimension}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise ValueError(f"Embedding {index} contains a non-finite value.")

        return self

    @property
    def count(self) -> int:
        return len(self.vectors)

    @classmethod
    def empty(cls, *, model: str) -> EmbeddingBatchResult:
        return cls(model=model, vectors=[], dimension=0)


class ItemEmbeddingDocument(StrictModel):
    """Canonical semantic document linked to one SQL receipt-item row."""

    item_id: int = Field(gt=0)
    receipt_id: int | None = Field(default=None, gt=0)
    description: str = Field(min_length=1, max_length=2000)
    normalized_description: str | None = Field(default=None, max_length=2000)
    category: str | None = Field(default=None, max_length=500)
    semantic_description: str | None = Field(default=None, max_length=2000)
    merchant: str | None = Field(default=None, max_length=500)
    parser_item_type: str | None = Field(default=None, max_length=100)
    text: str = Field(min_length=1, max_length=6000)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ItemEmbeddingIndexReport(StrictModel):
    """Summary returned by one incremental item-indexing run."""

    index_name: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=200)
    eligible_items: int = Field(ge=0)
    embedded: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    failed: int = Field(ge=0)
    pruned: int = Field(default=0, ge=0)
    dimension: int | None = Field(default=None, gt=0)
    batches: int = Field(default=0, ge=0)
    last_indexed_item_id: int | None = Field(default=None, gt=0)
    errors: list[str] = Field(default_factory=list)


class SemanticItemMatch(StrictModel):
    """One deduplicated approved purchase-product identity."""

    rank: int = Field(default=0, ge=0)
    item_id: int = Field(gt=0)
    item_ids: list[int] = Field(default_factory=list)
    occurrence_count: int = Field(default=1, ge=1)
    receipt_id: int = Field(gt=0)
    description: str = Field(min_length=1, max_length=2000)
    normalized_description: str | None = Field(default=None, max_length=2000)
    category: str | None = Field(default=None, max_length=500)
    semantic_description: str | None = Field(default=None, max_length=2000)
    merchant: str | None = Field(default=None, max_length=500)
    parser_item_type: str | None = Field(default=None, max_length=100)
    line_total: float | None = None
    unit_price: float | None = None
    receipt_date: str | None = Field(default=None, max_length=50)
    currency: str | None = Field(default=None, max_length=20)
    similarity: float = Field(ge=-1.0, le=1.0)
    vector_rank: int | None = Field(default=None, ge=1)
    lexical_rank: int | None = Field(default=None, ge=1)
    lexical_score: float = Field(default=0.0, ge=0.0)
    fusion_score: float = Field(default=0.0, ge=0.0)
    retrieval_method: str = Field(default="hybrid_rrf", max_length=100)

    @model_validator(mode="after")
    def validate_identity_group(self) -> Self:
        if not self.item_ids:
            self.item_ids = [self.item_id]
        if self.item_id not in self.item_ids:
            raise ValueError("item_id must be included in item_ids.")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("item_ids must not contain duplicates.")
        if self.occurrence_count != len(self.item_ids):
            raise ValueError("occurrence_count must equal len(item_ids).")
        return self


class SemanticItemSearchResult(StrictModel):
    """Complete result and diagnostics for one hybrid item query."""

    query: str = Field(min_length=1, max_length=2000)
    model: str = Field(min_length=1, max_length=200)
    dimension: int = Field(gt=0)
    total_candidates: int = Field(ge=0)
    raw_match_count: int = Field(default=0, ge=0)
    skipped_candidates: int = Field(default=0, ge=0)
    minimum_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    limit: int = Field(gt=0, le=1000)
    retrieval_mode: str = Field(default="hybrid_rrf", max_length=100)
    deduplicated: bool = True
    rrf_k: int = Field(default=60, ge=1)
    vector_weight: float = Field(default=1.0, ge=0.0)
    lexical_weight: float = Field(default=1.5, ge=0.0)
    ollama_calls: list[OllamaCallMetrics] = Field(default_factory=list, max_length=20)
    matches: list[SemanticItemMatch] = Field(default_factory=list)

    @property
    def returned_count(self) -> int:
        return len(self.matches)


class RetrievalEvaluationCase(StrictModel):
    """One expected-behavior case for a real semantic item index."""

    case_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=100)
    minimum_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    expected_item_ids: list[int] = Field(default_factory=list)
    expected_match_mode: str = Field(default="any", pattern=r"^(any|all)$")
    expected_any_terms: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    forbidden_parser_item_types: list[str] = Field(
        default_factory=lambda: ["discount", "deposit", "refund", "fee"]
    )
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def require_one_expectation(self) -> Self:
        if not self.expected_item_ids and not self.expected_any_terms:
            raise ValueError(
                "A retrieval evaluation case requires expected_item_ids or expected_any_terms."
            )
        return self


class RetrievalEvaluationCaseResult(StrictModel):
    """Measured result for one retrieval evaluation case."""

    case_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=2000)
    passed: bool
    top_k: int = Field(ge=1)
    returned_item_ids: list[int] = Field(default_factory=list)
    returned_identity_count: int = Field(default=0, ge=0)
    top_descriptions: list[str] = Field(default_factory=list)
    recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    precision_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    reciprocal_rank: float | None = Field(default=None, ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)


class RetrievalEvaluationReport(StrictModel):
    """Aggregate quality report for a retrieval regression corpus."""

    case_count: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    mean_reciprocal_rank: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_precision_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    results: list[RetrievalEvaluationCaseResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.passed + self.failed != self.case_count:
            raise ValueError("passed + failed must equal case_count.")
        if len(self.results) != self.case_count:
            raise ValueError("results length must equal case_count.")
        return self
