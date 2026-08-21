from __future__ import annotations

import json
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
        "extraction_json",
        "draft_json",
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
        queue_json = connection.execute(
            "SELECT extraction_json, draft_json, raw_json FROM review_queue WHERE job_id='job-1'"
        ).fetchone()

    assert revision["revision"] == 1
    assert queue is not None
    assert queue["review_revision"] == 1
    assert queue["receipt"]["merchant"]["name"] == "REWE Markt"
    assert queue["reviewer"] == "FT"
    assert len(history) == 1
    assert history[0]["revision"] == 1
    assert json.loads(queue_json["extraction_json"])["merchant"]["name"] == "REWE"
    assert json.loads(queue_json["draft_json"])["merchant"]["name"] == "REWE Markt"
    assert json.loads(queue_json["raw_json"])["merchant"]["name"] == "REWE"


def test_review_queue_upsert_preserves_extraction_snapshot(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    original = _receipt("REWE")
    database.upsert_review_queue(
        job_id="job-source",
        receipt=original,
        decision="review",
        balanced=True,
        difference=0.0,
        issue_count=1,
        image_path=None,
        final_receipt_path=None,
        queue_status="needs_review",
    )

    reviewed = _receipt("REWE Markt")
    reviewed["human_review"] = {
        "status": "needs_review",
        "reviewer": "FT",
    }
    database.upsert_review_queue(
        job_id="job-source",
        receipt=reviewed,
        decision="review",
        balanced=True,
        difference=0.0,
        issue_count=1,
        image_path=None,
        final_receipt_path=None,
        queue_status="needs_review",
    )

    with database.connect() as connection:
        row = connection.execute(
            "SELECT extraction_json, draft_json, raw_json "
            "FROM review_queue WHERE job_id='job-source'"
        ).fetchone()

    assert json.loads(row["extraction_json"])["merchant"]["name"] == "REWE"
    assert json.loads(row["draft_json"])["merchant"]["name"] == "REWE Markt"
    assert json.loads(row["raw_json"])["merchant"]["name"] == "REWE"


def test_migration_11_backfills_existing_queue_json(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    receipt = _receipt("Legacy Queue")
    database.upsert_review_queue(
        job_id="job-backfill",
        receipt=receipt,
        decision="review",
        balanced=True,
        difference=0.0,
        issue_count=1,
        image_path=None,
        final_receipt_path=None,
        queue_status="needs_review",
    )

    with database.connect() as connection:
        connection.execute(
            "UPDATE review_queue "
            "SET extraction_json=NULL, draft_json=NULL "
            "WHERE job_id='job-backfill'"
        )
        connection.execute("DELETE FROM schema_migrations WHERE version=11")
        connection.execute("UPDATE schema_meta SET value='10' WHERE key='schema_version'")
        connection.commit()

    database.migrations.migrate()

    with database.connect() as connection:
        row = connection.execute(
            "SELECT extraction_json, draft_json, raw_json "
            "FROM review_queue WHERE job_id='job-backfill'"
        ).fetchone()

    assert row["extraction_json"] == row["raw_json"]
    assert row["draft_json"] == row["raw_json"]


def test_migration_12_freezes_legacy_raw_json_without_losing_current_draft(
    tmp_path: Path,
) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    original = _receipt("REWE")
    database.upsert_review_queue(
        job_id="job-migrate-authority",
        receipt=original,
        decision="review",
        balanced=True,
        difference=0.0,
        issue_count=1,
        image_path=None,
        final_receipt_path=None,
        queue_status="needs_review",
    )
    corrected = _receipt("REWE Markt")
    database.save_review_revision(
        job_id="job-migrate-authority",
        receipt=corrected,
        requested_status="needs_review",
        effective_status="needs_review",
        reviewer="FT",
        notes=None,
        changed_fields=["merchant.name"],
        receipt_db_id=None,
    )

    with database.connect() as connection:
        # Emulate the migration-11 behavior where raw_json was still mutated
        # alongside draft_json.
        connection.execute(
            "UPDATE review_queue SET raw_json=draft_json WHERE job_id='job-migrate-authority'"
        )
        connection.execute("DELETE FROM schema_migrations WHERE version=12")
        connection.execute("UPDATE schema_meta SET value='11' WHERE key='schema_version'")
        connection.commit()

    database.migrations.migrate()

    with database.connect() as connection:
        row = connection.execute(
            "SELECT extraction_json, draft_json, raw_json "
            "FROM review_queue WHERE job_id='job-migrate-authority'"
        ).fetchone()

    assert json.loads(row["extraction_json"])["merchant"]["name"] == "REWE"
    assert json.loads(row["raw_json"])["merchant"]["name"] == "REWE"
    assert json.loads(row["draft_json"])["merchant"]["name"] == "REWE Markt"


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


def _enqueue_automatic_receipt(
    database: ReceiptDatabase,
    *,
    job_id: str,
    receipt: dict,
) -> dict:
    return database.upsert_review_queue(
        job_id=job_id,
        receipt=receipt,
        decision="import",
        balanced=True,
        difference=0.0,
        issue_count=0,
        image_path=None,
        final_receipt_path=None,
    )


def test_unknown_item_category_prevents_auto_validation(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    receipt = _receipt("Sergent Major")
    receipt["validation"]["issues"] = []
    receipt["categorization"] = {
        "status": "ok",
        "category_review_count": 1,
    }
    receipt["items"] = [
        {
            "description": "PLOTO PETTEN Naturel",
            "line_total": 5.0,
            "category_key": "unknown",
            "category_review_required": True,
        }
    ]

    result = _enqueue_automatic_receipt(
        database,
        job_id="job-unknown-category",
        receipt=receipt,
    )
    queue = database.get_review_queue_record("job-unknown-category")

    assert result["queue_status"] == "needs_review"
    assert result["reason_codes"] == [
        "CATEGORY_REVIEW_REQUIRED",
        "UNKNOWN_ITEM_CATEGORY",
    ]
    assert queue is not None
    assert queue["queue_status"] == "needs_review"
    assert queue["reason_codes"] == result["reason_codes"]


def test_explicit_category_review_flag_prevents_auto_validation(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    receipt = _receipt()
    receipt["validation"]["issues"] = []
    receipt["categorization"] = {
        "status": "ok",
        "category_review_count": 1,
    }
    receipt["items"] = [
        {
            "description": "Ambiguous product",
            "line_total": 5.0,
            "category_key": "clothing_shoes",
            "category_review_required": True,
        }
    ]

    result = _enqueue_automatic_receipt(
        database,
        job_id="job-category-review",
        receipt=receipt,
    )

    assert result["queue_status"] == "needs_review"
    assert result["reason_codes"] == ["CATEGORY_REVIEW_REQUIRED"]


def test_complete_item_categories_allow_auto_validation(tmp_path: Path) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    receipt = _receipt()
    receipt["validation"]["issues"] = []
    receipt["categorization"] = {
        "status": "ok",
        "category_review_count": 0,
    }
    receipt["items"] = [
        {
            "description": "T-Shirt",
            "line_total": 5.0,
            "category_key": "clothing_shoes",
            "category_review_required": False,
        }
    ]

    result = _enqueue_automatic_receipt(
        database,
        job_id="job-categorized",
        receipt=receipt,
    )

    assert result["queue_status"] == "auto_validated"
    assert result["reason_codes"] == []


def test_legacy_receipt_without_category_contract_keeps_previous_behavior(
    tmp_path: Path,
) -> None:
    database = ReceiptDatabase(tmp_path / "receipt.db")
    receipt = _receipt()
    receipt["validation"]["issues"] = []

    result = _enqueue_automatic_receipt(
        database,
        job_id="job-legacy",
        receipt=receipt,
    )

    assert result["queue_status"] == "auto_validated"
    assert result["reason_codes"] == []
