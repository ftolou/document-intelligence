from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from receipt_intelligence.rag.vector_codec import vector_to_blob
from receipt_intelligence.services.database_receipt_editor import DatabaseReceiptEditor
from receipt_intelligence.services.review_service import ReviewService
from receipt_intelligence.storage.job_store import JobStore
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def _receipt() -> dict:
    return {
        "merchant": {"name": "REWE"},
        "date": "2026-07-20",
        "currency": "EUR",
        "totals": {"grand_total": 5.0, "paid_total": 5.0},
        "items": [
            {
                "raw_description": "VITTEL 1,5L",
                "product_description": "VITTEL",
                "normalized_name": "vittel",
                "category": "item",
                "category_group": "Drinks",
                "category_key": "water",
                "line_total": 2.5,
            },
            {
                "raw_description": "ELMEX",
                "product_description": "ELMEX ZAHNPASTA",
                "normalized_name": "elmex zahnpasta",
                "category": "item",
                "category_group": "Personal Care",
                "category_key": "oral_care",
                "line_total": 2.5,
            },
        ],
        "human_review": {"status": "approved", "reviewer": "first"},
    }


def _seed_embeddings(database: ReceiptDatabase, item_ids: list[int]) -> None:
    with database.connect() as connection:
        for item_id in item_ids:
            connection.execute(
                """
                INSERT INTO rag_item_embeddings(
                    item_id, embedding_model, embedding_dimension,
                    document_text, content_hash, embedding, updated_at
                ) VALUES (?, 'test-model', 2, 'old', ?, ?, 'now')
                """,
                (item_id, f"{item_id:064x}"[-64:], sqlite3.Binary(vector_to_blob([1.0, 0.0]))),
            )
        connection.commit()


def test_database_editor_reindexes_only_product_text_changes(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    imported = database.import_receipt(job_id="job-1", receipt=_receipt())
    document = database.get_receipt_edit_document(imported.receipt_db_id)
    assert document is not None
    item_ids = [int(item["_db_item_id"]) for item in document["items"]]
    _seed_embeddings(database, item_ids)

    reindex_calls: list[list[int]] = []

    def reindex(item_ids_to_index: list[int]) -> dict:
        reindex_calls.append(list(item_ids_to_index))
        with database.connect() as connection:
            placeholders = ", ".join("?" for _ in item_ids_to_index)
            remaining = connection.execute(
                f"SELECT COUNT(*) AS n FROM rag_item_embeddings WHERE item_id IN ({placeholders})",
                item_ids_to_index,
            ).fetchone()["n"]
        assert remaining == 0
        return {"status": "current", "requested_item_ids": item_ids_to_index}

    editor = DatabaseReceiptEditor(
        database,
        ReviewService(JobStore(tmp_path / "jobs"), database),
        reindex_callback=reindex,
    )
    result = editor.save(
        imported.receipt_db_id,
        fields={"merchant_name": "REWE CITY"},
        item_corrections=[
            {"index": 0, "product_description": "VITTEL CLASSIC"},
            {"index": 1, "category_group": "Health", "category_key": "dental"},
        ],
        review={"status": "approved", "reviewer": "tester"},
    )

    assert reindex_calls == [item_ids]
    assert result["database_update"]["semantic_item_ids"] == item_ids
    assert result["database_update"]["metadata_item_ids"] == []
    assert result["semantic_index"]["status"] == "current"
    assert result["receipt"]["merchant"]["name"] == "REWE CITY"
    assert result["receipt"]["items"][0]["product_description"] == "VITTEL CLASSIC"
    assert result["receipt"]["items"][1]["category_key"] == "dental"

    with database.connect() as connection:
        remaining_ids = [
            int(row["item_id"])
            for row in connection.execute(
                "SELECT item_id FROM rag_item_embeddings ORDER BY item_id"
            ).fetchall()
        ]
    assert remaining_ids == []


def test_category_only_edit_reindexes_affected_item(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    imported = database.import_receipt(job_id="job-2", receipt=_receipt())
    document = database.get_receipt_edit_document(imported.receipt_db_id)
    assert document is not None
    item_id = int(document["items"][0]["_db_item_id"])
    _seed_embeddings(database, [item_id])

    reindex_calls: list[list[int]] = []
    editor = DatabaseReceiptEditor(
        database,
        ReviewService(JobStore(tmp_path / "jobs"), database),
        reindex_callback=lambda ids: reindex_calls.append(ids) or {"status": "current"},
    )
    result = editor.save(
        imported.receipt_db_id,
        fields={},
        item_corrections=[
            {"index": 0, "category_group": "Beverages", "category_key": "mineral_water"},
            {"index": 1},
        ],
        review={"status": "approved"},
    )

    assert reindex_calls == [[item_id]]
    assert result["database_update"]["semantic_item_ids"] == [item_id]
    assert result["semantic_index"]["status"] == "current"
    with database.connect() as connection:
        row = connection.execute(
            "SELECT category FROM receipt_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        count = connection.execute(
            "SELECT COUNT(*) AS n FROM rag_item_embeddings WHERE item_id = ?",
            (item_id,),
        ).fetchone()["n"]
    assert row["category"] == "Beverages/mineral_water"
    assert count == 0


def test_embedding_failure_does_not_rollback_database_edit(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    imported = database.import_receipt(job_id="job-3", receipt=_receipt())

    def fail_reindex(_item_ids: list[int]) -> dict:
        raise RuntimeError("ollama unavailable")

    editor = DatabaseReceiptEditor(
        database,
        ReviewService(JobStore(tmp_path / "jobs"), database),
        reindex_callback=fail_reindex,
    )
    result = editor.save(
        imported.receipt_db_id,
        fields={},
        item_corrections=[
            {"index": 0, "product_description": "VITTEL NATURELLE"},
            {"index": 1},
        ],
        review={"status": "approved"},
    )

    assert result["semantic_index"]["status"] == "failed"
    assert "ollama unavailable" in result["semantic_index"]["error"]
    stored = database.get_receipt_edit_document(imported.receipt_db_id)
    assert stored is not None
    assert stored["items"][0]["product_description"] == "VITTEL NATURELLE"


def test_semantic_description_edit_reindexes_only_affected_item(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    imported = database.import_receipt(job_id="job-semantic", receipt=_receipt())
    document = database.get_receipt_edit_document(imported.receipt_db_id)
    assert document is not None
    item_ids = [int(item["_db_item_id"]) for item in document["items"]]
    _seed_embeddings(database, item_ids)

    reindex_calls: list[list[int]] = []
    editor = DatabaseReceiptEditor(
        database,
        ReviewService(JobStore(tmp_path / "jobs"), database),
        reindex_callback=lambda ids: reindex_calls.append(ids) or {"status": "current"},
    )
    result = editor.save(
        imported.receipt_db_id,
        fields={},
        item_corrections=[
            {
                "index": 0,
                "semantic_description": "Vittel is bottled mineral water.",
                "category_reason": "Reviewed as bottled water.",
            },
            {"index": 1},
        ],
        review={"status": "approved"},
    )

    assert reindex_calls == [[item_ids[0]]]
    assert result["database_update"]["semantic_item_ids"] == [item_ids[0]]
    stored = database.get_receipt_edit_document(imported.receipt_db_id)
    assert stored is not None
    assert stored["items"][0]["semantic_description"] == "Vittel is bottled mineral water."
    assert stored["items"][0]["category_reason"] == "Reviewed as bottled water."
    with database.connect() as connection:
        row = connection.execute(
            "SELECT semantic_description, category_reason FROM receipt_items WHERE id = ?",
            (item_ids[0],),
        ).fetchone()
    assert row["semantic_description"] == "Vittel is bottled mineral water."
    assert row["category_reason"] == "Reviewed as bottled water."


def test_first_approval_indexes_all_items_and_refreshes_validation_state(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    receipt = _receipt()
    receipt["human_review"] = {"status": "needs_review"}
    receipt["validation"] = {
        "import_decision": "reject",
        "issues": [{"code": "MISSING_MERCHANT", "severity": "medium"}],
    }
    imported = database.import_receipt(job_id="job-first-approval", receipt=receipt)
    database.upsert_review_queue(
        job_id="job-first-approval",
        receipt=receipt,
        decision="reject",
        balanced=False,
        difference=None,
        issue_count=1,
        image_path=None,
        final_receipt_path=None,
        queue_status="rejected",
    )
    document = database.get_receipt_edit_document(imported.receipt_db_id)
    assert document is not None
    item_ids = [int(item["_db_item_id"]) for item in document["items"]]

    reindex_calls: list[list[int]] = []
    editor = DatabaseReceiptEditor(
        database,
        ReviewService(JobStore(tmp_path / "jobs"), database),
        reindex_callback=lambda ids: reindex_calls.append(ids) or {"status": "current"},
    )
    result = editor.save(
        imported.receipt_db_id,
        fields={},
        item_corrections=[{"index": 0}, {"index": 1}],
        review={"status": "approved", "reviewer": "tester"},
    )

    assert reindex_calls == [item_ids]
    assert result["receipt"]["human_review"]["status"] == "approved"
    assert result["receipt"]["validation"]["import_decision"] == "import"
    assert result["receipt"]["validation"]["pre_review_import_decision"] == "reject"
    assert result["database_update"]["previous_review_status"] == "needs_review"
    assert result["semantic_index"]["status"] == "current"
    queue = database.list_review_queue(status="approved")
    assert len(queue) == 1
    assert queue[0]["decision"] == "import"
    assert queue[0]["balanced"] == 1
    assert json.loads(queue[0]["raw_json"])["validation"]["import_decision"] == "import"
