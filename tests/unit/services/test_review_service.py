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


def _reviewable_receipt(*, merchant_name: str | None, parse_status: str = "failed") -> dict:
    return {
        "schema_version": "v14_6_llm_receipt_1",
        "parse_status": parse_status,
        "merchant": {"name": merchant_name},
        "date": "2026-07-24",
        "currency": "EUR",
        "totals": {"grand_total": 5.0, "paid_total": 5.0},
        "items": [
            {
                "description": "TEST ITEM",
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
            "issues": [{"code": "MISSING_MERCHANT", "severity": "medium"}],
        },
    }


def _write_ocr_context(store, job_id: str) -> None:
    job_dir = store.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{job_id}_v14_ocr_context.json").write_text(
        """{
  "lines": [
    {"line_id": "line_001", "text": "24.07.2026", "confidence": 0.99},
    {"line_id": "line_002", "text": "TEST ITEM 5,00", "confidence": 0.99},
    {"line_id": "line_003", "text": "CARD 5,00", "confidence": 0.99}
  ],
  "layout_rows": [
    {"row_id": "row_001", "text": "TEST ITEM 5,00", "source_line_ids": ["line_002"]}
  ]
}""",
        encoding="utf-8",
    )


def test_finalize_human_review_replaces_stale_rejection_after_correction(tmp_path) -> None:
    from receipt_intelligence.services.review_service import ReviewService, apply_human_review
    from receipt_intelligence.storage.job_store import JobStore
    from receipt_intelligence.storage.receipt_db import ReceiptDatabase

    store = JobStore(tmp_path / "jobs")
    database = ReceiptDatabase(tmp_path / "receipt.db")
    service = ReviewService(store, database)
    _write_ocr_context(store, "job-reviewed")
    corrected, _ = apply_human_review(
        _reviewable_receipt(merchant_name=None),
        {"merchant_name": "REWE"},
        [],
        {"status": "approved", "reviewer": "tester"},
    )

    result = service.finalize_human_review(
        "job-reviewed",
        corrected,
        requested_status="approved",
    )

    assert result["effective_status"] == "approved"
    assert result["import_allowed"] is True
    assert result["receipt"]["validation"]["import_decision"] == "import"
    assert result["receipt"]["validation"]["pre_review_import_decision"] == "reject"
    assert "MISSING_MERCHANT" not in {
        issue["code"] for issue in result["receipt"]["validation"]["issues"]
    }
    assert result["receipt"]["human_review"]["original_parse_status"] == "failed"


def test_finalize_human_review_blocks_approval_when_core_data_is_still_missing(tmp_path) -> None:
    from receipt_intelligence.services.review_service import ReviewService, apply_human_review
    from receipt_intelligence.storage.job_store import JobStore
    from receipt_intelligence.storage.receipt_db import ReceiptDatabase

    store = JobStore(tmp_path / "jobs")
    database = ReceiptDatabase(tmp_path / "receipt.db")
    service = ReviewService(store, database)
    _write_ocr_context(store, "job-blocked")
    reviewed, _ = apply_human_review(
        _reviewable_receipt(merchant_name=None, parse_status="partial"),
        {},
        [],
        {"status": "approved"},
    )

    result = service.finalize_human_review(
        "job-blocked",
        reviewed,
        requested_status="approved",
    )

    assert result["effective_status"] == "needs_review"
    assert result["approval_blocked"] is True
    assert result["import_allowed"] is False
    assert "MISSING_MERCHANT" in result["receipt"]["human_review"]["blocking_issue_codes"]
