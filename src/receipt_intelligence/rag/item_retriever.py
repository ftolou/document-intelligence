"""Hybrid semantic retrieval over approved receipt-item embeddings."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from receipt_intelligence.rag.hybrid_scoring import (
    build_fts_query,
    lexical_relevance,
    product_identity_key,
    reciprocal_rank_fusion,
)
from receipt_intelligence.rag.item_documents import is_indexable_description
from receipt_intelligence.rag.models import (
    EmbeddingBatchResult,
    SemanticItemMatch,
    SemanticItemSearchResult,
)
from receipt_intelligence.rag.ports import SemanticSearchCandidate, SemanticSearchRepository


class EmbeddingClient(Protocol):
    model: str

    def embed(self, texts: list[str]) -> EmbeddingBatchResult: ...


@dataclass
class _Candidate:
    row: SemanticSearchCandidate
    vector_similarity: float
    lexical_score: float
    fts_rank: int | None


@dataclass
class _IdentityGroup:
    key: str
    members: list[_Candidate]
    representative: _Candidate
    item_ids: list[int]
    vector_similarity: float
    lexical_score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None
    fusion_score: float = 0.0


class ItemSemanticRetriever:
    """Retrieve approved purchase-item identities with hybrid rank fusion.

    Dense vectors provide semantic recall from the approved product name,
    reviewed category path, and reviewed semantic description. Product-name
    lexical scoring and SQLite FTS5 preserve exact/compound-word precision.
    Merchant, prices, quantities, and dates remain structured metadata.
    """

    def __init__(
        self,
        *,
        repository: SemanticSearchRepository,
        embedding_client: EmbeddingClient,
        maximum_limit: int = 100,
        approved_only: bool = True,
        deduplicate: bool = True,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        lexical_weight: float = 1.5,
    ) -> None:
        if maximum_limit <= 0:
            raise ValueError("maximum_limit must be positive.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")
        if vector_weight < 0 or lexical_weight < 0:
            raise ValueError("retrieval weights must be non-negative.")
        if vector_weight == 0 and lexical_weight == 0:
            raise ValueError("at least one retrieval weight must be positive.")

        self.repository = repository
        self.embedding_client = embedding_client
        self.maximum_limit = int(maximum_limit)
        self.approved_only = bool(approved_only)
        self.deduplicate = bool(deduplicate)
        self.rrf_k = int(rrf_k)
        self.vector_weight = float(vector_weight)
        self.lexical_weight = float(lexical_weight)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        minimum_score: float | None = None,
        merchant: str | None = None,
        category: str | None = None,
        item_ids: Iterable[int] | None = None,
        deduplicate: bool | None = None,
    ) -> SemanticItemSearchResult:
        """Return hybrid-ranked approved purchase-item identities."""

        normalized_query = " ".join(str(query or "").split()).strip()
        if not normalized_query:
            raise ValueError("query must not be empty.")
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if limit > self.maximum_limit:
            raise ValueError(f"limit {limit} exceeds configured maximum {self.maximum_limit}.")
        if minimum_score is not None and not -1.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between -1 and 1.")

        query_result = self.embedding_client.embed([normalized_query])
        if query_result.count != 1 or query_result.dimension <= 0:
            raise ValueError("Embedding provider returned no usable query vector.")

        query_vector = np.asarray(query_result.vectors[0], dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(query_norm) or query_norm <= 0:
            raise ValueError("Query embedding has zero or invalid magnitude.")

        rows = self._load_candidates(
            merchant=merchant,
            category=category,
            item_ids=item_ids,
        )
        fts_ranks = self._load_fts_ranks(
            normalized_query,
            merchant=merchant,
            category=category,
            item_ids=item_ids,
        )

        candidates: list[_Candidate] = []
        skipped_candidates = 0
        for row in rows:
            if not is_indexable_description(row.description):
                skipped_candidates += 1
                continue
            dimension = int(row.embedding_dimension)
            if dimension != query_result.dimension:
                raise ValueError(
                    "Stored embedding dimension does not match query embedding: "
                    f"stored={dimension}, query={query_result.dimension}."
                )

            if row.vector is None:
                skipped_candidates += 1
                continue
            vector = np.asarray(row.vector, dtype=np.float32)

            vector_norm = float(np.linalg.norm(vector))
            if not np.isfinite(vector_norm) or vector_norm <= 0:
                skipped_candidates += 1
                continue

            similarity = float(np.dot(query_vector, vector) / (query_norm * vector_norm))
            similarity = max(-1.0, min(1.0, similarity))
            lexical_score = lexical_relevance(
                normalized_query,
                str(row.description or ""),
                str(row.normalized_description or "") or None,
            )
            item_id = int(row.item_id)
            fts_rank = fts_ranks.get(item_id)
            if fts_rank is not None:
                lexical_score += 1.0 / fts_rank

            # A dense threshold controls only the vector branch. Exact lexical
            # candidates remain eligible even when their vector score is lower.
            if minimum_score is not None and similarity < minimum_score and lexical_score <= 0:
                continue

            candidates.append(
                _Candidate(
                    row=row,
                    vector_similarity=similarity,
                    lexical_score=lexical_score,
                    fts_rank=fts_rank,
                )
            )

        use_deduplication = self.deduplicate if deduplicate is None else bool(deduplicate)
        groups = self._group_candidates(candidates, deduplicate=use_deduplication)
        self._rank_groups(groups)
        groups.sort(
            key=lambda group: (
                -group.fusion_score,
                -group.lexical_score,
                -group.vector_similarity,
                group.representative.row.item_id,
            )
        )

        matches = [
            self._group_to_match(group, rank=index)
            for index, group in enumerate(groups[:limit], start=1)
        ]

        return SemanticItemSearchResult(
            query=normalized_query,
            model=self.embedding_client.model,
            dimension=query_result.dimension,
            total_candidates=len(rows),
            raw_match_count=len(candidates),
            skipped_candidates=skipped_candidates,
            minimum_score=minimum_score,
            limit=limit,
            retrieval_mode="hybrid_rrf",
            deduplicated=use_deduplication,
            rrf_k=self.rrf_k,
            vector_weight=self.vector_weight,
            lexical_weight=self.lexical_weight,
            ollama_calls=query_result.ollama_calls,
            matches=matches,
        )

    def _group_candidates(
        self,
        candidates: list[_Candidate],
        *,
        deduplicate: bool,
    ) -> list[_IdentityGroup]:
        grouped: dict[str, list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            item_id = int(candidate.row.item_id)
            key = (
                product_identity_key(
                    str(candidate.row.description or ""),
                    str(candidate.row.normalized_description or "") or None,
                    fallback_item_id=item_id,
                )
                if deduplicate
                else f"item:{item_id}"
            )
            grouped[key].append(candidate)

        output: list[_IdentityGroup] = []
        for key, members in grouped.items():
            representative = max(
                members,
                key=lambda candidate: (
                    candidate.lexical_score,
                    candidate.vector_similarity,
                    -int(candidate.row.item_id),
                ),
            )
            output.append(
                _IdentityGroup(
                    key=key,
                    members=members,
                    representative=representative,
                    item_ids=sorted(int(member.row.item_id) for member in members),
                    vector_similarity=max(member.vector_similarity for member in members),
                    lexical_score=max(member.lexical_score for member in members),
                )
            )
        return output

    def _rank_groups(self, groups: list[_IdentityGroup]) -> None:
        for rank, group in enumerate(
            sorted(groups, key=lambda value: (-value.vector_similarity, value.key)),
            start=1,
        ):
            group.vector_rank = rank

        lexical_groups = [group for group in groups if group.lexical_score > 0]
        for rank, group in enumerate(
            sorted(
                lexical_groups,
                key=lambda value: (-value.lexical_score, -value.vector_similarity, value.key),
            ),
            start=1,
        ):
            group.lexical_rank = rank

        for group in groups:
            group.fusion_score = reciprocal_rank_fusion(
                vector_rank=group.vector_rank,
                lexical_rank=group.lexical_rank,
                rrf_k=self.rrf_k,
                vector_weight=self.vector_weight,
                lexical_weight=self.lexical_weight,
            )

    @staticmethod
    def _group_to_match(group: _IdentityGroup, *, rank: int) -> SemanticItemMatch:
        row = group.representative.row
        return SemanticItemMatch(
            rank=rank,
            item_id=int(row.item_id),
            item_ids=group.item_ids,
            occurrence_count=len(group.item_ids),
            receipt_id=int(row.receipt_id),
            description=str(row.description or ""),
            normalized_description=(
                str(row.normalized_description) if row.normalized_description else None
            ),
            category=_reviewed_category_from_raw_json(
                row.item_raw_json,
                fallback=str(row.category) if row.category else None,
            ),
            semantic_description=_semantic_description_from_raw_json(row.item_raw_json),
            merchant=str(row.merchant) if row.merchant else None,
            parser_item_type=(str(row.parser_item_type) if row.parser_item_type else None),
            line_total=float(row.line_total) if row.line_total is not None else None,
            unit_price=float(row.unit_price) if row.unit_price is not None else None,
            receipt_date=str(row.receipt_date) if row.receipt_date else None,
            currency=str(row.currency) if row.currency else None,
            similarity=group.vector_similarity,
            vector_rank=group.vector_rank,
            lexical_rank=group.lexical_rank,
            lexical_score=group.lexical_score,
            fusion_score=group.fusion_score,
            retrieval_method="hybrid_rrf",
        )

    def _load_candidates(
        self,
        *,
        merchant: str | None,
        category: str | None,
        item_ids: Iterable[int] | None,
    ) -> list[SemanticSearchCandidate]:
        return self.repository.load_candidates(
            embedding_model=self.embedding_client.model,
            approved_only=self.approved_only,
            merchant=merchant,
            category=category,
            item_ids=self._selected_item_ids(item_ids),
        )

    def _load_fts_ranks(
        self,
        query: str,
        *,
        merchant: str | None,
        category: str | None,
        item_ids: Iterable[int] | None,
    ) -> dict[int, int]:
        fts_query = build_fts_query(query)
        if not fts_query:
            return {}
        return self.repository.load_fts_ranks(
            fts_query=fts_query,
            approved_only=self.approved_only,
            merchant=merchant,
            category=category,
            item_ids=self._selected_item_ids(item_ids),
            maximum_results=max(self.maximum_limit * 10, 200),
        )

    @staticmethod
    def _selected_item_ids(item_ids: Iterable[int] | None) -> list[int] | None:
        selected = sorted({int(item_id) for item_id in item_ids or () if int(item_id) > 0})
        return selected or None


def _raw_item_payload(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _reviewed_category_from_raw_json(
    value: object,
    *,
    fallback: str | None,
) -> str | None:
    payload = _raw_item_payload(value)
    group = " ".join(str(payload.get("category_group") or "").split())
    key = " ".join(str(payload.get("category_key") or "").split())
    explicit = " ".join(
        str(
            payload.get("category_path")
            or payload.get("product_category")
            or payload.get("spending_category")
            or payload.get("analytics_category")
            or ""
        ).split()
    )
    if explicit:
        return explicit[:500]
    if group and key:
        return f"{group} / {key}"[:500]
    return (key or group or " ".join(str(fallback or "").split()))[:500] or None


def _semantic_description_from_raw_json(value: object) -> str | None:
    """Extract the approved semantic description from one item JSON payload."""

    payload = _raw_item_payload(value)
    description = " ".join(
        str(payload.get("semantic_description") or payload.get("category_reason") or "").split()
    )
    return description[:2000] or None
