from __future__ import annotations

import json
from pathlib import Path

from receipt_intelligence.application.use_cases.reviews import ReviewUseCases
from receipt_intelligence.services.database_receipt_editor import DatabaseReceiptEditor
from receipt_intelligence.services.review_service import ReviewService, apply_human_review
from receipt_intelligence.services.semantic_index_service import SemanticIndexUpdater
from receipt_intelligence.storage.job_store import JobStore
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def _receipt() -> dict:
    return {
        "schema_version": "v14_6_llm_receipt_1",
        "parse_status": "failed",
        "merchant": {"name": None},
        "date": "2026-07-24",
        "currency": "EUR",
        "totals": {"grand_total": 5.0, "paid_total": 5.0},
        "items": [
            {
                "description": "TEST ITEM",
                "product_description": "TEST ITEM",
                "normalized_name": "test item",
                "category": "item",
                "line_total": 5.0,
                "source_line_ids": ["line_002"],
            }
        ],
        "payments": [
            {"method": "card", "amount": 5.0, "source_line_ids": ["line_003"]}
        ],
        "validation": {
            "import_decision": "reject",
            "balanced": False,
            "difference": None,
            "issues": [{"code": "MISSING_MERCHANT", "severity": "medium"}],
        },
    }


def test_artifact_review_revalidates_imports_and_indexes_after_approval(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    database = ReceiptDatabase(tmp_path / "receipt.db")
    job_id = "job-artifact-review"
    job_dir = store.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    image_path = job_dir / "receipt.jpg"
    image_path.write_bytes(b"image")
    final_path = job_dir / f"{job_id}_receipt_final.json"
    final_path.write_text(json.dumps(_receipt()), encoding="utf-8")
    (job_dir / f"{job_id}_v14_ocr_context.json").write_text(
        json.dumps(
            {
                "lines": [
                    {"line_id": "line_001", "text": "24.07.2026", "confidence": 0.99},
                    {"line_id": "line_002", "text": "TEST ITEM 5,00", "confidence": 0.99},
                    {"line_id": "line_003", "text": "CARD 5,00", "confidence": 0.99},
                ],
                "layout_rows": [
                    {
                        "row_id": "row_001",
                        "text": "TEST ITEM 5,00",
                        "source_line_ids": ["line_002"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store.create(job_id, {"image_path": str(image_path)})

    indexed: list[list[int]] = []
    updater = SemanticIndexUpdater(
        database,
        reindex_callback=lambda ids: indexed.append(ids) or {"status": "current"},
    )
    review_service = ReviewService(
        store,
        database,
        semantic_index_updater=updater,
    )
    editor = DatabaseReceiptEditor(database, review_service)
    use_cases = ReviewUseCases(
        store,
        database,
        review_service,
        editor,
        apply_human_review,
    )

    result = use_cases.save_review(
        job_id,
        fields={"merchant_name": "REWE"},
        item_corrections=[{"index": 0}],
        review={"status": "approved", "reviewer": "tester"},
    )

    assert result["receipt"]["human_review"]["status"] == "approved"
    assert result["receipt"]["validation"]["import_decision"] == "import"
    assert result["receipt"]["validation"]["pre_review_import_decision"] == "reject"
    assert result["receipt_db_import"]["receipt_db_id"] is not None
    assert result["semantic_index"]["status"] == "current"
    assert len(indexed) == 1
    assert len(indexed[0]) == 1

    queue = database.list_review_queue(status="approved")
    assert len(queue) == 1
    assert queue[0]["decision"] == "import"
    assert queue[0]["balanced"] == 1
    queued_receipt = json.loads(queue[0]["raw_json"])
    assert queued_receipt["merchant"]["name"] == "REWE"
    assert queued_receipt["validation"]["import_decision"] == "import"

    approved = json.loads(review_service.approved_receipt_path(job_id).read_text())
    assert approved["validation"]["import_decision"] == "import"
