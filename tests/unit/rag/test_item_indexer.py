from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from receipt_intelligence.adapters.storage.sqlite.semantic_index import (
    SQLiteSemanticIndexRepository,
)
from receipt_intelligence.rag.item_indexer import ItemEmbeddingIndexer
from receipt_intelligence.rag.models import EmbeddingBatchResult
from receipt_intelligence.rag.vector_codec import blob_to_vector, vector_to_blob
from receipt_intelligence.storage.connection import SQLiteConnectionFactory
from receipt_intelligence.storage.migrations import MigrationRunner


class FakeEmbeddingClient:
    model = "test-embedding-model"

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def embed(self, texts: list[str]) -> EmbeddingBatchResult:
        self.calls.append(list(texts))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("provider unavailable")
        vectors = [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]
        return EmbeddingBatchResult(model=self.model, vectors=vectors, dimension=2)


def _database(tmp_path: Path) -> Path:
    path = tmp_path / "receipt.db"
    factory = SQLiteConnectionFactory(path)
    MigrationRunner(factory).migrate()
    with factory.connect() as connection:
        connection.execute(
            """
            INSERT INTO receipts(
                id, job_id, merchant_name, merchant_normalized, review_status,
                approved_receipt_path, raw_json, created_at, updated_at
            ) VALUES
                (1, 'approved-job', 'REWE', 'rewe', 'approved', '/approved.json', '{}', 'now', 'now'),
                (2, 'draft-job', 'DM', 'dm', 'draft', NULL, '{}', 'now', 'now')
            """
        )
        connection.executemany(
            """
            INSERT INTO receipt_items(
                id, receipt_id, item_index, raw_name, normalized_name,
                category, parser_item_type, embedding_text, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    1,
                    1,
                    0,
                    "VITTEL",
                    "vittel",
                    "groceries_beverages",
                    "item",
                    "x",
                    '{"category_reason":"Vittel is a brand of mineral water."}',
                ),
                (2, 1, 1, "AKTIONSRABATT", "aktionsrabatt", "discount", "discount", "x", "{}"),
                (3, 1, 2, "DAMEN SNEAKER", "damen sneaker", "clothing/shoes", "item", "x", "{}"),
                (4, 2, 0, "ZAHNPASTA", "zahnpasta", "personal_care", "item", "x", "{}"),
            ],
        )
        connection.commit()
    return path


def test_incremental_indexer_embeds_only_approved_purchase_items(tmp_path: Path) -> None:
    database = _database(tmp_path)
    client = FakeEmbeddingClient()
    report = ItemEmbeddingIndexer(
        repository=SQLiteSemanticIndexRepository(database),
        embedding_client=client,
        batch_size=1,
    ).rebuild()

    assert report.eligible_items == 2
    assert report.embedded == 2
    assert report.unchanged == 0
    assert report.failed == 0
    assert len(client.calls) == 2

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT item_id, embedding_dimension, embedding FROM rag_item_embeddings ORDER BY item_id"
        ).fetchall()
    assert [row[0] for row in rows] == [1, 3]
    assert blob_to_vector(rows[0][2], dimension=rows[0][1])[0] == pytest.approx(1.0)


def test_unchanged_hashes_are_skipped_and_changed_documents_are_reembedded(tmp_path: Path) -> None:
    database = _database(tmp_path)
    first_client = FakeEmbeddingClient()
    indexer = ItemEmbeddingIndexer(
        repository=SQLiteSemanticIndexRepository(database),
        embedding_client=first_client,
    )
    first = indexer.rebuild()
    second = indexer.rebuild()

    assert first.embedded == 2
    assert second.embedded == 0
    assert second.unchanged == 2
    assert len(first_client.calls) == 1

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE receipt_items SET raw_name='VITTEL CLASSIC' WHERE id=1")
        connection.commit()

    third = indexer.rebuild()
    assert third.embedded == 1
    assert third.unchanged == 1
    assert len(first_client.calls) == 2


def test_category_or_reviewed_reason_changes_reembed_but_merchant_does_not(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    client = FakeEmbeddingClient()
    indexer = ItemEmbeddingIndexer(
        repository=SQLiteSemanticIndexRepository(database),
        embedding_client=client,
    )
    first = indexer.rebuild()

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE receipts SET merchant_name='OTHER STORE' WHERE id=1")
        connection.commit()
    merchant_only = indexer.rebuild()

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE receipt_items SET category='beverages/water' WHERE id=1")
        connection.commit()
    category_changed = indexer.rebuild()

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE receipt_items SET raw_json=? WHERE id=1",
            ('{"category_reason":"Still mineral water, now sparkling."}',),
        )
        connection.commit()
    reason_changed = indexer.rebuild()

    assert first.embedded == 2
    assert merchant_only.embedded == 0
    assert merchant_only.unchanged == 2
    assert category_changed.embedded == 1
    assert reason_changed.embedded == 1


def test_force_reembeds_all_eligible_rows(tmp_path: Path) -> None:
    database = _database(tmp_path)
    client = FakeEmbeddingClient()
    indexer = ItemEmbeddingIndexer(
        repository=SQLiteSemanticIndexRepository(database),
        embedding_client=client,
    )
    indexer.rebuild()
    report = indexer.rebuild(force=True)
    assert report.embedded == 2
    assert report.unchanged == 0


def test_batch_failure_is_recorded_and_later_batches_continue(tmp_path: Path) -> None:
    database = _database(tmp_path)
    client = FakeEmbeddingClient(fail_on_call=1)
    report = ItemEmbeddingIndexer(
        repository=SQLiteSemanticIndexRepository(database),
        embedding_client=client,
        batch_size=1,
    ).rebuild()

    assert report.embedded == 1
    assert report.failed == 1
    assert report.errors
    with sqlite3.connect(database) as connection:
        state = connection.execute(
            "SELECT indexed_count, failed_count, last_error FROM rag_index_state"
        ).fetchone()
    assert state[0] == 1
    assert state[1] == 1
    assert "provider unavailable" in state[2]


def test_index_item_ids_limits_scope(tmp_path: Path) -> None:
    database = _database(tmp_path)
    client = FakeEmbeddingClient()
    report = ItemEmbeddingIndexer(
        repository=SQLiteSemanticIndexRepository(database),
        embedding_client=client,
    ).index_item_ids([3])
    assert report.eligible_items == 1
    assert report.embedded == 1
    with sqlite3.connect(database) as connection:
        ids = [row[0] for row in connection.execute("SELECT item_id FROM rag_item_embeddings")]
    assert ids == [3]


def test_rebuild_prunes_previously_indexed_placeholder_rows(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE receipt_items SET raw_name = 'Product Purchase', normalized_name = 'product purchase' WHERE id = 1"
        )
        connection.execute(
            """
            INSERT INTO rag_item_embeddings(
                item_id, embedding_model, embedding_dimension,
                document_text, content_hash, embedding, updated_at
            ) VALUES (1, 'test-embedding-model', 2, 'old placeholder', ?, ?, 'now')
            """,
            ("f" * 64, sqlite3.Binary(vector_to_blob([1.0, 0.0]))),
        )
        connection.commit()

    report = ItemEmbeddingIndexer(
        repository=SQLiteSemanticIndexRepository(database),
        embedding_client=FakeEmbeddingClient(),
    ).rebuild()

    assert report.pruned == 1
    with sqlite3.connect(database) as connection:
        remaining = connection.execute(
            "SELECT COUNT(*) FROM rag_item_embeddings WHERE item_id = 1"
        ).fetchone()[0]
    assert remaining == 0
