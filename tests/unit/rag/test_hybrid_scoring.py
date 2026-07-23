from __future__ import annotations

import pytest

from receipt_intelligence.rag.hybrid_scoring import (
    build_fts_query,
    lexical_relevance,
    product_identity_key,
    reciprocal_rank_fusion,
)


def test_lexical_relevance_handles_german_compounds_without_dictionary() -> None:
    assert lexical_relevance("Wasser", "MINERALWASSER") > 0
    assert lexical_relevance("Schuhe", "HS-Halbschuhe") > 0
    assert lexical_relevance("Schuhe", "KRAWATTE") == 0


def test_identity_key_uses_normalized_description() -> None:
    assert (
        product_identity_key(
            "KRAWATTE",
            "Krawatte",
            fallback_item_id=3,
        )
        == "krawatte"
    )


def test_rrf_rewards_lexical_and_dense_agreement() -> None:
    hybrid = reciprocal_rank_fusion(vector_rank=2, lexical_rank=1)
    dense_only = reciprocal_rank_fusion(vector_rank=1, lexical_rank=None)
    assert hybrid > dense_only


def test_rrf_validates_configuration() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion(vector_rank=1, lexical_rank=1, rrf_k=0)


def test_fts_query_is_limited_to_product_name_columns() -> None:
    query = build_fts_query('Wasser "Classic"')
    assert query is not None
    assert "raw_name:" in query
    assert "normalized_name:" in query
    assert "merchant" not in query
    assert "category" not in query
