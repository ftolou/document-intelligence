"""Human-review queue and duplicate-candidate persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from receipt_intelligence.storage.fingerprints import (
    duplicate_score_against_row,
    file_sha256,
    receipt_core,
)
from receipt_intelligence.storage.normalization import utc_now
from receipt_intelligence.storage.repositories.base import BaseRepository


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
        if queue_status is None:
            if duplicate_status:
                queue_status = "duplicate_candidate"
            elif decision in {"reject", "llm_failed"}:
                queue_status = "rejected"
            elif decision in {"import", "ok", "auto_validated"} and balanced is True:
                queue_status = "auto_validated"
            else:
                queue_status = "needs_review"

        now = utc_now()
        raw_json = json.dumps(receipt, ensure_ascii=False, default=str)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO review_queue(
                    job_id, queue_status, decision, balanced, difference, issue_count,
                    merchant_name, merchant_normalized, receipt_date, receipt_time,
                    grand_total, item_count, file_sha256, content_fingerprint,
                    item_signature, image_path, final_receipt_path, duplicate_status,
                    duplicate_score, duplicate_candidates_json, raw_json, created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    file_sha256=excluded.file_sha256,
                    content_fingerprint=excluded.content_fingerprint,
                    item_signature=excluded.item_signature,
                    image_path=excluded.image_path,
                    final_receipt_path=excluded.final_receipt_path,
                    duplicate_status=excluded.duplicate_status,
                    duplicate_score=excluded.duplicate_score,
                    duplicate_candidates_json=excluded.duplicate_candidates_json,
                    raw_json=excluded.raw_json,
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
        }

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
            "WHEN queue_status='rejected' THEN 2 ELSE 3 END, "
            "updated_at DESC LIMIT ?"
        )
        parameters.append(max(1, min(500, int(limit))))
        with self.connect() as connection:
            rows: list[dict[str, Any]] = []
            for row in connection.execute(sql, parameters).fetchall():
                item = dict(row)
                try:
                    item["duplicate_candidates"] = json.loads(
                        item.get("duplicate_candidates_json") or "[]"
                    )
                except Exception:
                    item["duplicate_candidates"] = []
                rows.append(item)
            return rows

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
            connection.execute(
                """
                UPDATE review_queue
                SET queue_status=?, receipt_db_id=COALESCE(?, receipt_db_id),
                    updated_at=?
                WHERE job_id=?
                """,
                (status, receipt_db_id, now, job_id),
            )
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
