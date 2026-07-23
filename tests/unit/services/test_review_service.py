"""Regression tests for the extracted human-review service."""

from __future__ import annotations

from receipt_intelligence.services.review_service import apply_human_review


def test_apply_human_review_keeps_parser_and_spending_categories_separate() -> None:
    receipt = {
        "merchant": {"name": "OLD"},
        "totals": {"grand_total": 10.0},
        "items": [
            {
                "description": "MILCH",
                "category": "item",
                "category_group": "Food",
                "category_key": "Other",
            }
        ],
    }

    updated, changed = apply_human_review(
        receipt,
        {"merchant_name": "REWE", "grand_total": "12.50"},
        [
            {
                "index": 0,
                "parser_item_type": "discount",
                "category_group": "Food",
                "category_key": "Dairy",
            }
        ],
        {"status": "approved", "reviewer": "tester"},
    )

    assert updated["merchant"]["name"] == "REWE"
    assert updated["totals"]["grand_total"] == 12.5
    assert updated["items"][0]["category"] == "discount"
    assert updated["items"][0]["parser_item_type"] == "discount"
    assert updated["items"][0]["category_path"] == "Food/Dairy"
    assert updated["human_review"]["status"] == "approved"
    assert "items[0].category" in changed


def test_preferred_receipt_path_uses_approved_data_before_original(tmp_path) -> None:
    from receipt_intelligence.services.review_service import ReviewService
    from receipt_intelligence.storage.job_store import JobStore
    from receipt_intelligence.storage.receipt_db import ReceiptDatabase

    store = JobStore(tmp_path / "jobs")
    database = ReceiptDatabase(tmp_path / "receipt.db")
    service = ReviewService(store, database)
    job_dir = store.job_dir("job-1")
    job_dir.mkdir(parents=True, exist_ok=True)
    final_path = job_dir / "receipt_final.json"
    approved_path = job_dir / "approved_receipt.json"
    final_path.write_text('{"merchant":{"name":"ORIGINAL"}}', encoding="utf-8")
    approved_path.write_text('{"merchant":{"name":"APPROVED"}}', encoding="utf-8")

    selected = service.preferred_receipt_path("job-1")

    assert selected == approved_path.resolve()


def test_safe_job_artifact_path_rejects_files_outside_job_directory(tmp_path) -> None:
    from receipt_intelligence.services.review_service import ReviewService
    from receipt_intelligence.storage.job_store import JobStore
    from receipt_intelligence.storage.receipt_db import ReceiptDatabase

    store = JobStore(tmp_path / "jobs")
    database = ReceiptDatabase(tmp_path / "receipt.db")
    service = ReviewService(store, database)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    assert service.safe_job_artifact_path("job-1", outside) is None


def test_import_reviewed_receipt_preserves_database_image_without_job_status(tmp_path) -> None:
    from receipt_intelligence.services.review_service import ReviewService
    from receipt_intelligence.storage.job_store import JobStore
    from receipt_intelligence.storage.receipt_db import ReceiptDatabase

    store = JobStore(tmp_path / "jobs")
    database = ReceiptDatabase(tmp_path / "receipt.db")
    service = ReviewService(store, database)
    job_dir = store.job_dir("job-image")
    job_dir.mkdir(parents=True, exist_ok=True)
    image_path = job_dir / "receipt.jpg"
    image_path.write_bytes(b"image")
    approved_path = job_dir / "approved_receipt.json"
    receipt = {
        "merchant": {"name": "LIDL"},
        "currency": "EUR",
        "totals": {"grand_total": 5.10},
        "items": [],
        "human_review": {"status": "approved"},
    }
    approved_path.write_text("{}", encoding="utf-8")
    imported = database.import_receipt(
        job_id="job-image",
        receipt=receipt,
        approved_receipt_path=approved_path,
        image_path=image_path,
    )

    service.import_reviewed_receipt("job-image", receipt, approved_path)
    stored = database.get_receipt_review_record(imported.receipt_db_id)

    assert stored is not None
    assert stored["image_path"] == str(image_path)
