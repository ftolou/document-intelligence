"""Persistence contracts used by semantic indexing and retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class IndexableItemSource:
    """Storage-neutral source fields required to build one embedding document."""

    item_id: int
    receipt_id: int | None
    description: str | None
    description_normalized: str | None
    category: str | None
    category_group: str | None
    category_key: str | None
    category_reason: str | None
    semantic_description: str | None
    item_raw_json: object
    merchant: str | None
    parser_item_type: str | None

    def as_mapping(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "receipt_id": self.receipt_id,
            "description": self.description,
            "description_normalized": self.description_normalized,
            "category": self.category,
            "category_group": self.category_group,
            "category_key": self.category_key,
            "category_reason": self.category_reason,
            "semantic_description": self.semantic_description,
            "item_raw_json": self.item_raw_json,
            "merchant": self.merchant,
            "parser_item_type": self.parser_item_type,
        }


@dataclass(frozen=True, slots=True)
class StoredItemEmbedding:
    """One encoded vector prepared for durable storage."""

    item_id: int
    embedding_model: str
    embedding_dimension: int
    document_text: str
    content_hash: str
    vector: tuple[float, ...]
    updated_at: str


@dataclass(frozen=True, slots=True)
class SemanticIndexState:
    """Progress snapshot for one semantic index."""

    index_name: str
    embedding_model: str
    embedding_dimension: int | None
    indexed_count: int
    failed_count: int
    last_indexed_item_id: int | None
    last_completed_at: str
    last_error: str | None


@dataclass(frozen=True, slots=True)
class SemanticSearchCandidate:
    """Storage-neutral candidate row used by hybrid retrieval."""

    item_id: int
    embedding_dimension: int
    vector: tuple[float, ...] | None
    receipt_id: int
    description: str | None
    normalized_description: str | None
    category: str | None
    item_raw_json: object
    parser_item_type: str | None
    line_total: float | None
    unit_price: float | None
    merchant: str | None
    receipt_date: str | None
    currency: str | None


class SemanticIndexRepository(Protocol):
    """Persistence operations required by the incremental embedding indexer."""

    def load_indexable_items(
        self,
        *,
        approved_only: bool,
        item_ids: Sequence[int] | None = None,
    ) -> list[IndexableItemSource]: ...

    def prune_embeddings(
        self,
        *,
        embedding_model: str,
        eligible_item_ids: set[int],
        scope_item_ids: set[int] | None = None,
    ) -> int: ...

    def existing_hashes(
        self,
        *,
        embedding_model: str,
        item_ids: Sequence[int] | None = None,
    ) -> dict[int, str]: ...

    def known_dimension(self, *, embedding_model: str) -> int | None: ...

    def store_embeddings(self, records: Sequence[StoredItemEmbedding]) -> None: ...

    def save_state(self, state: SemanticIndexState) -> None: ...


class SemanticSearchRepository(Protocol):
    """Read-side persistence operations required by hybrid item retrieval."""

    def load_candidates(
        self,
        *,
        embedding_model: str,
        approved_only: bool,
        merchant: str | None,
        category: str | None,
        item_ids: Sequence[int] | None,
    ) -> list[SemanticSearchCandidate]: ...

    def load_fts_ranks(
        self,
        *,
        fts_query: str,
        approved_only: bool,
        merchant: str | None,
        category: str | None,
        item_ids: Sequence[int] | None,
        maximum_results: int,
    ) -> dict[int, int]: ...


__all__ = [
    "IndexableItemSource",
    "SemanticIndexRepository",
    "SemanticIndexState",
    "SemanticSearchCandidate",
    "SemanticSearchRepository",
    "StoredItemEmbedding",
]
