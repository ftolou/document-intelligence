"""Human-review queue, canonical review drafts, and review history persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from receipt_intelligence.receipt_compat import (
    validation_for_review,
    validation_issues,
)
from receipt_intelligence.storage.fingerprints import (
    duplicate_score_against_row,
    file_sha256,
    receipt_core,
)
from receipt_intelligence.storage.normalization import utc_now
from receipt_intelligence.storage.repositories.base import BaseRepository


def _json_list(value: Any) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _category_review_reason_codes(receipt: dict[str, Any]) -> set[str]:
    categorization = (
        receipt.get("categorization") if isinstance(receipt.get("categorization"), dict) else {}
    )
    items = receipt.get("items") if isinstance(receipt.get("items"), list) else []
    category_items = [item for item in items if isinstance(item, dict)]
    has_category_contract = bool(categorization) or any(
        "category_key" in item or "category_review_required" in item for item in category_items
    )
    if not has_category_contract:
        return set()

    codes: set[str] = set()
    status = str(categorization.get("status") or "").strip().lower()
    if categorization and status != "ok":
        codes.add("CATEGORIZATION_INCOMPLETE")

    declared_review_count = categorization.get("category_review_count")
    if isinstance(declared_review_count, (int, float)) and declared_review_count > 0:
        codes.add("CATEGORY_REVIEW_REQUIRED")

    for item in category_items:
        category_key = str(item.get("category_key") or "").strip().lower()
        if item.get("category_review_required") is True:
            codes.add("CATEGORY_REVIEW_REQUIRED")
        if not category_key or category_key == "unknown":
            codes.add("UNKNOWN_ITEM_CATEGORY")

    return codes


def _review_reason_codes(receipt: dict[str, Any]) -> list[str]:
    validation = receipt.get("validation") if isinstance(receipt.get("validation"), dict) else {}
    issues = validation_issues(validation)
    codes = {
        str(issue.get("code") or "").strip()
        for issue in issues
        if isinstance(issue, dict) and str(issue.get("code") or "").strip()
    }
    codes.update(_category_review_reason_codes(receipt))
    human_review = (
        receipt.get("human_review") if isinstance(receipt.get("human_review"), dict) else {}
    )
    codes.update(
        str(code).strip()
        for code in (human_review.get("blocking_issue_codes") or [])
        if str(code).strip()
    )
    return sorted(codes)


class ReviewRepository(BaseRepository):
    def find_duplicate_candidates(
        self,
        *,
        job_id: str,
        receipt: dict[str, Any],
        image_path: Path | str | None = None,
        threshold: float = 70.0,
    ) -> list[dict[str, Any]]:
        core = receipt_core(receipt)
        image_hash = file_sha256(image_path)
        candidates: list[dict[str, Any]] = []
        with self.connect() as connection:
            approved_rows = connection.execute(
                """
                SELECT id AS receipt_db_id, job_id, merchant_normalized,
                       receipt_date, receipt_time, grand_total, file_sha256,
                       content_fingerprint, raw_json
                FROM receipts
                WHERE job_id IS NULL OR job_id != ?
                """,
                (job_id,),
            ).fetchall()
            for row in approved_rows:
                score, reasons = duplicate_score_against_row(core, image_hash, row)
                if score >= threshold:
                    candidates.append(
                        {
                            "candidate_type": "approved_receipt",
                            "candidate_receipt_db_id": row["receipt_db_id"],
                            "candidate_job_id": row["job_id"],
                            "score": round(score, 2),
                            "reasons": reasons,
                        }
                    )

            queued_rows = connection.execute(
                """
                SELECT job_id, merchant_normalized, receipt_date, receipt_time,
                       grand_total, file_sha256, content_fingerprint, item_signature
                FROM review_queue
                WHERE job_id != ?
                """,
                (job_id,),
            ).fetchall()
            for row in queued_rows:
                score, reasons = duplicate_score_against_row(core, image_hash, row)
                if score >= threshold:
                    candidates.append(
                        {
                            "candidate_type": "review_queue",
                            "candidate_job_id": row["job_id"],
                            "score": round(score, 2),
                            "reasons": reasons,
                        }
                    )
        candidates.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
        return candidates[:10]

    def upsert_review_queue(
        self,
        *,
        job_id: str,
        receipt: dict[str, Any],
        decision: str | None,
        balanced: bool | None,
        difference: float | None,
        issue_count: int | None,
        image_path: Path | str | None,
        final_receipt_path: Path | str | None,
        queue_status: str | None = None,
    ) -> dict[str, Any]:
        core = receipt_core(receipt)
        image_hash = file_sha256(image_path)
        validation = (
            receipt.get("validation") if isinstance(receipt.get("validation"), dict) else {}
        )
        review_validation = validation_for_review(validation)
        current_issues = validation_issues(validation)
        decision = decision or review_validation.get("import_decision")
        if balanced is None:
            balanced = review_validation.get("balanced")
        if difference is None:
            difference = review_validation.get("difference")
        issue_count = len(current_issues)
        duplicates = self.find_duplicate_candidates(
            job_id=job_id,
            receipt=receipt,
            image_path=image_path,
        )
        max_score = max(
            [float(candidate.get("score") or 0) for candidate in duplicates],
            default=0.0,
        )
        duplicate_status = "duplicate_candidate" if max_score >= 70 else None
        category_reason_codes = _category_review_reason_codes(receipt)
        reason_codes = _review_reason_codes(receipt)
        if queue_status is None:
            if duplicate_status:
                queue_status = "duplicate_candidate"
            elif decision in {"reject", "llm_failed"}:
                queue_status = "rejected"
            elif (
                decision in {"import", "ok", "auto_validated"}
                and balanced is True
                and not category_reason_codes
            ):
                queue_status = "auto_validated"
            else:
                queue_status = "needs_review"

        now = utc_now()
        raw_json = json.dumps(receipt, ensure_ascii=False, default=str)
        human_review = (
            receipt.get("human_review") if isinstance(receipt.get("human_review"), dict) else {}
        )
        source_kind = "reviewed" if human_review else "extraction"
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO review_queue(
                    job_id, queue_status, decision, balanced, difference, issue_count,
                    merchant_name, merchant_normalized, receipt_date, receipt_time,
                    grand_total, item_count, file_sha256, content_fingerprint,
                    item_signature, image_path, final_receipt_path, duplicate_status,
                    duplicate_score, duplicate_candidates_json, raw_json,
                    reviewer, review_notes, reviewed_at, review_reason_codes_json,
                    source_kind, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    queue_status=excluded.queue_status,
                    decision=excluded.decision,
                    balanced=excluded.balanced,
                    difference=excluded.difference,
                    issue_count=excluded.issue_count,
                    merchant_name=excluded.merchant_name,
                    merchant_normalized=excluded.merchant_normalized,
                    receipt_date=excluded.receipt_date,
                    receipt_time=excluded.receipt_time,
                    grand_total=excluded.grand_total,
                    item_count=excluded.item_count,
                    file_sha256=COALESCE(excluded.file_sha256, review_queue.file_sha256),
                    content_fingerprint=excluded.content_fingerprint,
                    item_signature=excluded.item_signature,
                    image_path=COALESCE(excluded.image_path, review_queue.image_path),
                    final_receipt_path=COALESCE(excluded.final_receipt_path, review_queue.final_receipt_path),
                    duplicate_status=excluded.duplicate_status,
                    duplicate_score=excluded.duplicate_score,
                    duplicate_candidates_json=excluded.duplicate_candidates_json,
                    raw_json=excluded.raw_json,
                    reviewer=COALESCE(excluded.reviewer, review_queue.reviewer),
                    review_notes=COALESCE(excluded.review_notes, review_queue.review_notes),
                    reviewed_at=COALESCE(excluded.reviewed_at, review_queue.reviewed_at),
                    review_reason_codes_json=excluded.review_reason_codes_json,
                    source_kind=excluded.source_kind,
                    updated_at=excluded.updated_at
                """,
                (
                    job_id,
                    queue_status,
                    decision,
                    1 if balanced else 0 if balanced is False else None,
                    difference,
                    issue_count,
                    core["merchant_name"],
                    core["merchant_normalized"],
                    core["receipt_date"],
                    core["receipt_time"],
                    core["grand_total"],
                    core["item_count"],
                    image_hash,
                    core["content_fingerprint"],
                    core["item_signature"],
                    str(image_path) if image_path else None,
                    str(final_receipt_path) if final_receipt_path else None,
                    duplicate_status,
                    max_score,
                    json.dumps(duplicates, ensure_ascii=False, default=str),
                    raw_json,
                    str(human_review.get("reviewer") or "") or None,
                    str(human_review.get("notes") or "") or None,
                    str(human_review.get("reviewed_at") or "") or None,
                    json.dumps(reason_codes, ensure_ascii=False),
                    source_kind,
                    now,
                    now,
                ),
            )
            connection.execute("DELETE FROM duplicate_candidates WHERE job_id = ?", (job_id,))
            for duplicate in duplicates:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO duplicate_candidates(
                        job_id, candidate_job_id, candidate_receipt_db_id, score,
                        status, reason_json, created_at
                    ) VALUES (?, ?, ?, ?, 'candidate', ?, ?)
                    """,
                    (
                        job_id,
                        duplicate.get("candidate_job_id"),
                        duplicate.get("candidate_receipt_db_id"),
                        float(duplicate.get("score") or 0),
                        json.dumps(duplicate, ensure_ascii=False, default=str),
                        now,
                    ),
                )
            connection.commit()
        return {
            "job_id": job_id,
            "queue_status": queue_status,
            "duplicate_status": duplicate_status,
            "duplicate_score": round(max_score, 2),
            "duplicate_candidates": duplicates,
            "reason_codes": reason_codes,
        }

    def get_review_queue_record(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_queue WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        if row is None:
            return None
        return self._present_queue_row(dict(row))

    def list_review_queue(
        self,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM review_queue"
        parameters: list[Any] = []
        if status and status != "all":
            sql += " WHERE queue_status = ? OR duplicate_status = ?"
            parameters.extend([status, status])
        sql += (
            " ORDER BY CASE "
            "WHEN queue_status='duplicate_candidate' THEN 0 "
            "WHEN queue_status='needs_review' THEN 1 "
            "WHEN queue_status='rejected' THEN 2 "
            "WHEN queue_status='auto_validated' THEN 3 ELSE 4 END, "
            "updated_at ASC LIMIT ?"
        )
        parameters.append(max(1, min(500, int(limit))))
        with self.connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._present_queue_row(dict(row)) for row in rows]

    def review_queue_summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT queue_status, COUNT(*) AS n
                FROM review_queue
                GROUP BY queue_status
                """
            ).fetchall()
            total = int(
                connection.execute("SELECT COUNT(*) AS n FROM review_queue").fetchone()["n"]
            )
        counts = {str(row["queue_status"]): int(row["n"]) for row in rows}
        return {
            "total": total,
            "needs_review": counts.get("needs_review", 0),
            "duplicate_candidate": counts.get("duplicate_candidate", 0),
            "rejected": counts.get("rejected", 0),
            "auto_validated": counts.get("auto_validated", 0),
            "approved": counts.get("approved", 0),
            "imported": counts.get("imported", 0),
            "counts": counts,
        }

    def save_review_revision(
        self,
        *,
        job_id: str,
        receipt: dict[str, Any],
        requested_status: str | None,
        effective_status: str,
        reviewer: str | None,
        notes: str | None,
        changed_fields: list[str],
        receipt_db_id: int | None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        validation = (
            receipt.get("validation") if isinstance(receipt.get("validation"), dict) else {}
        )
        reason_codes = _review_reason_codes(receipt)
        receipt_json = json.dumps(receipt, ensure_ascii=False, default=str)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT review_revision FROM review_queue WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is None:
                raise KeyError("review queue record not found")
            current_revision = int(row["review_revision"] or 0)
            if expected_revision is not None and current_revision != int(expected_revision):
                raise ValueError("review state is stale; reload the receipt before saving")
            revision = current_revision + 1
            cursor = connection.execute(
                """
                UPDATE review_queue
                SET review_revision=?, queue_status=?, receipt_db_id=COALESCE(?, receipt_db_id),
                    reviewer=?, review_notes=?, reviewed_at=?, review_reason_codes_json=?,
                    source_kind='reviewed', raw_json=?, updated_at=?
                WHERE job_id=? AND review_revision=?
                """,
                (
                    revision,
                    effective_status,
                    receipt_db_id,
                    reviewer,
                    notes,
                    now,
                    json.dumps(reason_codes, ensure_ascii=False),
                    receipt_json,
                    now,
                    str(job_id),
                    current_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("review state is stale; reload the receipt before saving")
            connection.execute(
                """
                INSERT INTO receipt_review_history(
                    job_id, receipt_db_id, revision, requested_status, effective_status,
                    reviewer, notes, changed_fields_json, validation_json,
                    receipt_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(job_id),
                    receipt_db_id,
                    revision,
                    requested_status,
                    effective_status,
                    reviewer,
                    notes,
                    json.dumps(sorted(set(changed_fields)), ensure_ascii=False),
                    json.dumps(validation, ensure_ascii=False, default=str),
                    receipt_json,
                    now,
                ),
            )
            connection.commit()
        return {
            "job_id": str(job_id),
            "receipt_db_id": receipt_db_id,
            "revision": revision,
            "queue_status": effective_status,
            "reviewed_at": now,
            "reason_codes": reason_codes,
        }

    def update_review_status(
        self,
        job_id: str,
        status: str,
        *,
        receipt_db_id: int | None = None,
    ) -> dict[str, Any]:
        allowed = {
            "needs_review",
            "approved",
            "auto_validated",
            "rejected",
            "duplicate_candidate",
            "duplicate_confirmed",
            "dismissed_duplicate",
            "imported",
        }
        if status not in allowed:
            raise ValueError(f"Unsupported review status: {status}")
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE review_queue
                SET queue_status=?, receipt_db_id=COALESCE(?, receipt_db_id),
                    updated_at=?
                WHERE job_id=?
                """,
                (status, receipt_db_id, now, job_id),
            )
            if cursor.rowcount == 0:
                raise KeyError("review queue record not found")
            if status == "duplicate_confirmed":
                connection.execute(
                    """
                    UPDATE duplicate_candidates
                    SET status='confirmed_duplicate', resolved_at=?
                    WHERE job_id=? AND status='candidate'
                    """,
                    (now, job_id),
                )
            elif status == "dismissed_duplicate":
                connection.execute(
                    """
                    UPDATE duplicate_candidates
                    SET status='dismissed', resolved_at=?
                    WHERE job_id=? AND status='candidate'
                    """,
                    (now, job_id),
                )
            connection.commit()
        return {"job_id": job_id, "queue_status": status, "updated_at": now}

    @staticmethod
    def _present_queue_row(item: dict[str, Any]) -> dict[str, Any]:
        item["duplicate_candidates"] = _json_list(item.get("duplicate_candidates_json"))
        receipt = _json_object(item.get("raw_json"))
        core = receipt_core(receipt)
        validation = (
            receipt.get("validation") if isinstance(receipt.get("validation"), dict) else {}
        )
        review_validation = validation_for_review(validation)
        issues = validation_issues(validation)
        item.update(
            merchant_name=core.get("merchant_name"),
            merchant_normalized=core.get("merchant_normalized"),
            receipt_date=core.get("receipt_date"),
            receipt_time=core.get("receipt_time"),
            grand_total=core.get("grand_total"),
            item_count=core.get("item_count"),
            decision=review_validation.get("import_decision") or item.get("decision"),
            balanced=review_validation.get("balanced"),
            difference=review_validation.get("difference"),
            issue_count=len(issues),
        )
        item["reason_codes"] = _review_reason_codes(receipt)
        item["receipt"] = receipt
        return item
