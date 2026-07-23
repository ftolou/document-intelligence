"""Deterministic lexical and rank-fusion helpers for hybrid item retrieval."""

from __future__ import annotations

from collections.abc import Iterable

from receipt_intelligence.storage.normalization import normalize_text


def product_identity_key(
    description: str,
    normalized_description: str | None,
    *,
    fallback_item_id: int,
) -> str:
    """Return a stable identity key for retrieval-result deduplication."""

    identity = normalize_text(normalized_description or description)
    return identity or f"item:{fallback_item_id}"


def lexical_relevance(
    query: str,
    description: str,
    normalized_description: str | None = None,
) -> float:
    """Score literal product-name overlap without merchant/category leakage.

    The scorer intentionally works only on product-identifying text. It handles
    German compounds such as ``Mineralwasser`` for the query ``Wasser`` and
    ``Halbschuhe`` for ``Schuhe`` without maintaining a product dictionary.
    """

    normalized_query = normalize_text(query)
    normalized_text = normalize_text(
        " ".join(part for part in (description, normalized_description or "") if part)
    )
    if not normalized_query or not normalized_text:
        return 0.0

    query_tokens = _unique_tokens(normalized_query.split())
    text_tokens = _unique_tokens(normalized_text.split())
    if not query_tokens or not text_tokens:
        return 0.0

    score = 0.0
    if normalized_query == normalized_text:
        score += 20.0
    elif normalized_query in normalized_text:
        score += 12.0 + min(2.0, len(normalized_query) / max(len(normalized_text), 1))
    elif normalized_text in normalized_query:
        score += 5.0

    for query_token in query_tokens:
        best_token_score = 0.0
        for text_token in text_tokens:
            if query_token == text_token:
                candidate = 8.0
            elif len(query_token) >= 4 and query_token in text_token:
                candidate = 6.0 + min(1.0, len(query_token) / len(text_token))
            elif len(text_token) >= 4 and text_token in query_token:
                candidate = 4.0 + min(1.0, len(text_token) / len(query_token))
            else:
                candidate = _character_ngram_score(query_token, text_token)
            best_token_score = max(best_token_score, candidate)
        score += best_token_score

    return round(score, 8)


def reciprocal_rank_fusion(
    *,
    vector_rank: int | None,
    lexical_rank: int | None,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    lexical_weight: float = 1.5,
) -> float:
    """Combine dense and lexical ranks with weighted reciprocal-rank fusion."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")
    if vector_weight < 0 or lexical_weight < 0:
        raise ValueError("RRF weights must be non-negative.")

    score = 0.0
    if vector_rank is not None:
        score += vector_weight / (rrf_k + vector_rank)
    if lexical_rank is not None:
        score += lexical_weight / (rrf_k + lexical_rank)
    return score


def build_fts_query(query: str) -> str | None:
    """Build a safe FTS5 prefix query limited to product-name columns."""

    tokens = [token for token in _unique_tokens(normalize_text(query).split()) if len(token) >= 2]
    if not tokens:
        return None

    clauses: list[str] = []
    for token in tokens[:12]:
        escaped = token.replace('"', '""')
        clauses.append(f'raw_name:"{escaped}"*')
        clauses.append(f'normalized_name:"{escaped}"*')
    return " OR ".join(clauses)


def _unique_tokens(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            output.append(token)
    return output


def _character_ngram_score(left: str, right: str) -> float:
    if min(len(left), len(right)) < 4:
        return 0.0

    left_ngrams = _ngrams(left, 3)
    right_ngrams = _ngrams(right, 3)
    union = left_ngrams | right_ngrams
    if not union:
        return 0.0
    similarity = len(left_ngrams & right_ngrams) / len(union)
    if similarity < 0.45:
        return 0.0
    return 2.5 * similarity


def _ngrams(value: str, size: int) -> set[str]:
    if len(value) <= size:
        return {value}
    return {value[index : index + size] for index in range(len(value) - size + 1)}
