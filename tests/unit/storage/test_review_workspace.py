from __future__ import annotations

from pathlib import Path

from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def _receipt(name: str = "REWE") -> dict:
    return {
        "merchant": {"name": name},
        "date": "2026-07-26",
        "currency": "EUR",
        "totals": {"grand_total": 5.0, "paid_total": 5.0},
        "items": [{"description": "TEST", "line_total": 5.0}],
        "validation": {
            "import_decision": "review",
            "balanced": True,
            "difference": 0.0,
            "issues": [{"code": "CATEGORY_UNCERTAIN", "severity": "low"}],
        },
    }


def test_review_workspace_migration_adds_canonical_draft_and_history_schema(
    tmp_path: Path,
) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    with database.connect() as connection:
        queue_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(review_queue)")
        }
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert {
        "review_revision",
        "reviewer",
        "review_notes",
        "reviewed_at",
        "review_reason_codes_json",
        "source_kind",
    } <= queue_columns
    assert "receipt_review_history" in tables


def test_review_revision_updates_canonical_queue_draft_and_appends_history(
    tmp_path: Path,
) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    receipt = _receipt()
    database.upsert_review_queue(
        job_id="job-1",
        receipt=receipt,
        decision="review",
        balanced=True,
        difference=0.0,
        issue_count=1,
        image_path=None,
        final_receipt_path=None,
        queue_status="needs_review",
    )

    corrected = _receipt("REWE Markt")
    corrected["human_review"] = {
        "status": "needs_review",
        "reviewer": "FT",
        "notes": "Merchant corrected",
    }
    revision = database.save_review_revision(
        job_id="job-1",
        receipt=corrected,
        requested_status="needs_review",
        effective_status="needs_review",
        reviewer="FT",
        notes="Merchant corrected",
        changed_fields=["merchant.name"],
        receipt_db_id=None,
    )

    queue = database.get_review_queue_record("job-1")
    with database.connect() as connection:
        history = connection.execute(
            "SELECT * FROM receipt_review_history WHERE job_id='job-1'"
        ).fetchall()

    assert revision["revision"] == 1
    assert queue is not None
    assert queue["review_revision"] == 1
    assert queue["receipt"]["merchant"]["name"] == "REWE Markt"
    assert queue["reviewer"] == "FT"
    assert len(history) == 1
    assert history[0]["revision"] == 1


def test_review_revision_rejects_stale_expected_revision(tmp_path: Path) -> None:
    import pytest

    database = ReceiptDatabase(tmp_path / "receipt.db")
    receipt = _receipt()
    database.upsert_review_queue(
        job_id="job-stale",
        receipt=receipt,
        decision="review",
        balanced=True,
        difference=0.0,
        issue_count=1,
        image_path=None,
        final_receipt_path=None,
        queue_status="needs_review",
    )
    database.save_review_revision(
        job_id="job-stale",
        receipt=receipt,
        requested_status="needs_review",
        effective_status="needs_review",
        reviewer="A",
        notes=None,
        changed_fields=[],
        receipt_db_id=None,
        expected_revision=0,
    )

    with pytest.raises(ValueError, match="stale"):
        database.save_review_revision(
            job_id="job-stale",
            receipt=receipt,
            requested_status="approved",
            effective_status="approved",
            reviewer="B",
            notes=None,
            changed_fields=[],
            receipt_db_id=None,
            expected_revision=0,
        )
