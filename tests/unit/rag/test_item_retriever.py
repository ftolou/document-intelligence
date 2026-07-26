from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from receipt_intelligence.adapters.storage.sqlite.semantic_search import (
    SQLiteSemanticSearchRepository,
)
from receipt_intelligence.rag.item_retriever import ItemSemanticRetriever
from receipt_intelligence.rag.models import EmbeddingBatchResult
from receipt_intelligence.rag.vector_codec import vector_to_blob
from receipt_intelligence.storage.connection import SQLiteConnectionFactory
from receipt_intelligence.storage.migrations import MigrationRunner


class StaticEmbeddingClient:
    model = "test-model"

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> EmbeddingBatchResult:
        self.calls.append(list(texts))
        return EmbeddingBatchResult(
            model=self.model,
            vectors=[list(self.vector) for _ in texts],
            dimension=len(self.vector),
        )


def _database(tmp_path: Path) -> Path:
    path = tmp_path / "receipt.db"
    factory = SQLiteConnectionFactory(path)
    MigrationRunner(factory).migrate()
    with factory.connect() as connection:
        connection.execute(
            """
            INSERT INTO receipts(
                id, job_id, merchant_name, merchant_normalized, receipt_date,
                currency, review_status, approved_receipt_path, raw_json,
                created_at, updated_at
            ) VALUES
                (1, 'approved', 'Store', 'store', '2026-07-01', 'EUR',
                 'approved', '/approved.json', '{}', 'now', 'now'),
                (2, 'approved-2', 'Store', 'store', '2026-07-02', 'EUR',
                 'approved', '/approved2.json', '{}', 'now', 'now'),
                (3, 'draft', 'Draft Store', 'draft store', '2026-07-03', 'EUR',
                 'draft', NULL, '{}', 'now', 'now')
            """
        )
        connection.executemany(
            """
            INSERT INTO receipt_items(
                id, receipt_id, item_index, raw_name, normalized_name,
                category, parser_item_type, line_total, unit_price,
                embedding_text, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'x', ?)
            """,
            [
                (
                    1,
                    1,
                    0,
                    "SCHUHENGEL GRAU",
                    "schuhengel grau",
                    "clothing_shoes",
                    "item",
                    139.95,
                    139.95,
                    "{}",
                ),
                (2, 1, 1, "KRAWATTE", "krawatte", "clothing_shoes", "item", 14.24, 14.24, "{}"),
                (
                    3,
                    1,
                    2,
                    "MINERALWASSER",
                    "mineralwasser",
                    "groceries_beverages",
                    "item",
                    0.79,
                    0.79,
                    '{"category_reason":"Mineral water for drinking."}',
                ),
                (
                    4,
                    1,
                    3,
                    "DENKMIT WC BLAUSPÜLER",
                    "denkmit wc blauspueler",
                    "cleaning",
                    "item",
                    1.99,
                    1.99,
                    "{}",
                ),
                (5, 2, 0, "KRAWATTE", "krawatte", "clothing_shoes", "item", 15.00, 15.00, "{}"),
                (
                    6,
                    1,
                    4,
                    "AKTIONSRABATT",
                    "aktionsrabatt",
                    "discount",
                    "discount",
                    -1.0,
                    None,
                    "{}",
                ),
                (7, 3, 0, "DRAFT SHOE", "draft shoe", "clothing_shoes", "item", 10.0, None, "{}"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO rag_item_embeddings(
                item_id, embedding_model, embedding_dimension,
                document_text, content_hash, embedding, updated_at
            ) VALUES (?, 'test-model', 2, 'doc', ?, ?, 'now')
            """,
            [
                (1, "a" * 64, sqlite3.Binary(vector_to_blob([0.8, 0.2]))),
                (2, "b" * 64, sqlite3.Binary(vector_to_blob([1.0, 0.0]))),
                (3, "c" * 64, sqlite3.Binary(vector_to_blob([0.0, 1.0]))),
                (4, "d" * 64, sqlite3.Binary(vector_to_blob([0.98, 0.02]))),
                (5, "e" * 64, sqlite3.Binary(vector_to_blob([1.0, 0.0]))),
                (6, "f" * 64, sqlite3.Binary(vector_to_blob([0.9, 0.1]))),
                (7, "1" * 64, sqlite3.Binary(vector_to_blob([1.0, 0.0]))),
            ],
        )
        connection.commit()
    return path


def test_hybrid_ranking_promotes_literal_product_match_over_category_contamination(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    client = StaticEmbeddingClient([1.0, 0.0])

    result = ItemSemanticRetriever(
        repository=SQLiteSemanticSearchRepository(database),
        embedding_client=client,
    ).search("Schuhe", limit=5)

    assert result.matches[0].item_id == 1
    assert result.matches[0].description == "SCHUHENGEL GRAU"
    assert result.matches[0].lexical_rank == 1
    assert result.matches[0].retrieval_method == "hybrid_rrf"
    assert result.retrieval_mode == "hybrid_rrf"
    assert result.total_candidates == 5
    assert client.calls == [["Schuhe"]]


def test_compound_word_lexical_signal_recovers_mineralwasser(tmp_path: Path) -> None:
    database = _database(tmp_path)
    # Dense similarity intentionally favors the WC cleaner.
    result = ItemSemanticRetriever(
        repository=SQLiteSemanticSearchRepository(database),
        embedding_client=StaticEmbeddingClient([1.0, 0.0]),
    ).search("Wasser", limit=5)

    assert result.matches[0].item_id == 3
    assert result.matches[0].description == "MINERALWASSER"
    assert result.matches[0].semantic_description == "Mineral water for drinking."
    assert result.matches[0].lexical_score > 0


def test_search_deduplicates_product_occurrences_and_retains_all_item_ids(
    tmp_path: Path,
) -> None:
    result = ItemSemanticRetriever(
        repository=SQLiteSemanticSearchRepository(_database(tmp_path)),
        embedding_client=StaticEmbeddingClient([1.0, 0.0]),
    ).search("Krawatte", limit=10)

    krawatte = next(match for match in result.matches if match.description == "KRAWATTE")
    assert krawatte.item_ids == [2, 5]
    assert krawatte.occurrence_count == 2
    assert [match.description for match in result.matches].count("KRAWATTE") == 1


def test_rejected_receipt_with_stale_embedding_is_not_retrieved(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE receipts SET review_status='rejected', approved_receipt_path='/legacy.json' "
            "WHERE id=2"
        )
        connection.commit()

    result = ItemSemanticRetriever(
        repository=SQLiteSemanticSearchRepository(database),
        embedding_client=StaticEmbeddingClient([1.0, 0.0]),
    ).search("Krawatte", limit=10)

    krawatte = next(match for match in result.matches if match.description == "KRAWATTE")
    assert krawatte.item_ids == [2]
    assert krawatte.occurrence_count == 1
    assert [match.description for match in result.matches].count("KRAWATTE") == 1


def test_search_can_disable_deduplication(tmp_path: Path) -> None:
    result = ItemSemanticRetriever(
        repository=SQLiteSemanticSearchRepository(_database(tmp_path)),
        embedding_client=StaticEmbeddingClient([1.0, 0.0]),
    ).search("Krawatte", limit=10, deduplicate=False)

    assert [match.description for match in result.matches].count("KRAWATTE") == 2
    assert all(match.occurrence_count == 1 for match in result.matches)


def test_search_supports_score_and_structured_filters(tmp_path: Path) -> None:
    database = _database(tmp_path)
    retriever = ItemSemanticRetriever(
        repository=SQLiteSemanticSearchRepository(database),
        embedding_client=StaticEmbeddingClient([0.0, 1.0]),
    )

    result = retriever.search(
        "Wasser",
        limit=10,
        minimum_score=0.9,
        merchant="Store",
        category="groceries_beverages",
    )

    assert [match.item_id for match in result.matches] == [3]
    assert result.matches[0].line_total == 0.79
    assert result.matches[0].currency == "EUR"


def test_search_can_limit_candidates_by_item_id(tmp_path: Path) -> None:
    result = ItemSemanticRetriever(
        repository=SQLiteSemanticSearchRepository(_database(tmp_path)),
        embedding_client=StaticEmbeddingClient([1.0, 0.0]),
    ).search("anything", item_ids=[3], limit=10)

    assert [match.item_id for match in result.matches] == [3]


def test_search_rejects_dimension_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dimension does not match"):
        ItemSemanticRetriever(
            repository=SQLiteSemanticSearchRepository(_database(tmp_path)),
            embedding_client=StaticEmbeddingClient([1.0, 0.0, 0.0]),
        ).search("Schuhe")


def test_search_skips_corrupt_or_zero_vectors(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE rag_item_embeddings SET embedding = ? WHERE item_id = 1",
            (sqlite3.Binary(b"bad"),),
        )
        connection.execute(
            "UPDATE rag_item_embeddings SET embedding = ? WHERE item_id = 2",
            (sqlite3.Binary(vector_to_blob([0.0, 0.0])),),
        )
        connection.commit()

    result = ItemSemanticRetriever(
        repository=SQLiteSemanticSearchRepository(database),
        embedding_client=StaticEmbeddingClient([1.0, 0.0]),
    ).search("Schuhe")

    assert result.skipped_candidates == 2
    assert all(match.item_id not in {1, 2} for match in result.matches)


def test_search_validates_query_limit_threshold_and_configuration(tmp_path: Path) -> None:
    database = _database(tmp_path)
    retriever = ItemSemanticRetriever(
        repository=SQLiteSemanticSearchRepository(database),
        embedding_client=StaticEmbeddingClient([1.0, 0.0]),
        maximum_limit=5,
    )

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.search(" ")
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        retriever.search("shoe", limit=6)
    with pytest.raises(ValueError, match="between -1 and 1"):
        retriever.search("shoe", limit=5, minimum_score=1.2)
    with pytest.raises(ValueError, match="at least one retrieval weight"):
        ItemSemanticRetriever(
            repository=SQLiteSemanticSearchRepository(database),
            embedding_client=StaticEmbeddingClient([1.0, 0.0]),
            vector_weight=0,
            lexical_weight=0,
        )
