"""Human-review and review-queue endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from receipt_intelligence.services.artifact_service import artifact_url
from receipt_intelligence.services.database_receipt_editor import DatabaseReceiptEditor
from receipt_intelligence.services.review_service import ReviewService, apply_human_review
from receipt_intelligence.web.dependencies import get_app_services

review_bp = Blueprint("review", __name__)


def _review_service() -> ReviewService:
    services = get_app_services()
    return ReviewService(services.job_store, services.receipt_db)


def _source_label(path: Path) -> str:
    return "approved_receipt" if path.name == "approved_receipt.json" else "final_receipt"


def _review_artifacts(
    job_id: str,
    *,
    job: dict[str, Any] | None,
    review_service: ReviewService,
    database_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = (
        job.get("result") if isinstance(job, dict) and isinstance(job.get("result"), dict) else {}
    )
    existing = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    artifacts = dict(existing)
    image_url = None
    if database_record:
        image_url = review_service.database_image_url(database_record)
    if not image_url:
        image_url = review_service.review_image_url(job_id)
    if image_url:
        artifacts["receipt_image"] = image_url
    return artifacts


def _review_payload_from_job(job_id: str) -> tuple[dict[str, Any] | None, str | None]:
    services = get_app_services()
    store = services.job_store
    review_service = _review_service()
    database_record = services.receipt_db.get_receipt_review_record_by_job_id(job_id)
    if database_record and database_record.get("id") is not None:
        try:
            return (
                DatabaseReceiptEditor(services.receipt_db, review_service).load(
                    int(database_record["id"])
                ),
                None,
            )
        except Exception as exc:
            return None, f"could not read database receipt: {exc}"
    source_path = review_service.preferred_receipt_path(
        job_id,
        stored_approved_path=(database_record or {}).get("approved_receipt_path"),
        stored_source_path=(database_record or {}).get("source_receipt_path"),
    )
    if source_path is None:
        return None, "approved/final receipt artifact not found"
    try:
        receipt = review_service.read_receipt_json(source_path)
    except Exception as exc:
        return None, f"could not read receipt JSON: {exc}"

    job = store.get(job_id)
    artifacts = _review_artifacts(
        job_id,
        job=job,
        review_service=review_service,
        database_record=database_record,
    )
    return (
        {
            "job_id": job_id,
            "receipt_db_id": (database_record or {}).get("id"),
            "receipt": receipt,
            "review": review_service.load_review_record(job_id, receipt),
            "artifacts": artifacts,
            "receipt_image": artifacts.get("receipt_image"),
            "source": _source_label(source_path),
            "editable": True,
            "save_url": f"/api/review/{job_id}",
            "read_only_reason": None,
        },
        None,
    )


@review_bp.get("/api/review/<job_id>")
def get_human_review(job_id: str):
    payload, error = _review_payload_from_job(job_id)
    if payload is None:
        return jsonify({"error": error or "review source unavailable"}), 404
    return jsonify(payload)


@review_bp.post("/api/review/<job_id>")
def save_human_review(job_id: str):
    services = get_app_services()
    store = services.job_store
    receipt_db = services.receipt_db
    review_service = _review_service()

    database_record = receipt_db.get_receipt_review_record_by_job_id(job_id)
    if database_record and database_record.get("id") is not None:
        payload = request.get_json(silent=True) or {}
        fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
        item_corrections = payload.get("items") if isinstance(payload.get("items"), list) else []
        review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
        try:
            return jsonify(
                DatabaseReceiptEditor(receipt_db, review_service).save(
                    int(database_record["id"]),
                    fields=fields,
                    item_corrections=item_corrections,
                    review=review,
                )
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
    stored_source_path = review_service.safe_job_artifact_path(
        job_id,
        (database_record or {}).get("source_receipt_path"),
    )
    original_source_path = stored_source_path or review_service.final_receipt_path(job_id)
    source_path = review_service.preferred_receipt_path(
        job_id,
        stored_approved_path=(database_record or {}).get("approved_receipt_path"),
        stored_source_path=(database_record or {}).get("source_receipt_path"),
    )
    if source_path is None:
        return jsonify({"error": "approved/final receipt artifact not found"}), 404

    payload = request.get_json(silent=True) or {}
    fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
    item_corrections = payload.get("items") if isinstance(payload.get("items"), list) else []
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}

    try:
        receipt = review_service.read_receipt_json(source_path)
        approved, changed = apply_human_review(
            receipt,
            fields,
            item_corrections,
            review,
        )
        approved_path = review_service.approved_receipt_path(job_id)
        review_path = review_service.review_record_path(job_id)
        approved_path.parent.mkdir(parents=True, exist_ok=True)
        approved_path.write_text(
            json.dumps(approved, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        record = {
            "job_id": job_id,
            "source_receipt": (original_source_path or source_path).name,
            "approved_receipt": approved_path.name,
            "status": approved.get("human_review", {}).get("status"),
            "reviewer": approved.get("human_review", {}).get("reviewer"),
            "notes": approved.get("human_review", {}).get("notes"),
            "reviewed_at": approved.get("human_review", {}).get("reviewed_at"),
            "changed_fields": changed,
            "submitted_fields": fields,
            "submitted_items": item_corrections,
        }
        review_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        job = store.get(job_id)
        if job is not None:
            store.register_artifact(job_id, "approved_receipt", approved_path, category="review")
            store.register_artifact(job_id, "human_review", review_path, category="review")

        db_import = review_service.import_reviewed_receipt(
            job_id,
            approved,
            approved_path,
            original_source_path or source_path,
        )
        record["receipt_db_import"] = db_import
        queue_status = (
            "approved"
            if record.get("status") not in {"rejected", "duplicate_confirmed"}
            else str(record.get("status"))
        )
        receipt_db.update_review_status(
            job_id,
            queue_status,
            receipt_db_id=db_import.get("receipt_db_id"),
        )
        review_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if job is not None:
            store.register_artifact(job_id, "human_review", review_path, category="review")

        artifacts = _review_artifacts(
            job_id,
            job=job,
            review_service=review_service,
            database_record=receipt_db.get_receipt_review_record_by_job_id(job_id),
        )
        artifacts["approved_receipt"] = artifact_url(job_id, approved_path)
        artifacts["human_review"] = artifact_url(job_id, review_path)
        if job is not None:
            result = job.get("result") if isinstance(job.get("result"), dict) else {}
            result["artifacts"] = artifacts
            store.update(
                job_id,
                result=result,
                review=record,
                receipt_db_import=db_import,
            )
            store.add_event(
                job_id,
                {
                    "stage": "human_review",
                    "status": "done",
                    "message": f"Human review saved with {len(changed)} changed field(s).",
                    "details": {
                        "review_status": record.get("status"),
                        "changed_fields": changed,
                    },
                },
            )
        return jsonify(
            {
                "ok": True,
                "job_id": job_id,
                "receipt_db_id": db_import.get("receipt_db_id"),
                "review": record,
                "receipt": approved,
                "artifacts": artifacts,
                "receipt_db_import": db_import,
                "source": "approved_receipt",
                "editable": True,
                "save_url": f"/api/review/{job_id}",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@review_bp.get("/api/review-queue")
def review_queue_list():
    status_filter = request.args.get("status") or "all"
    try:
        limit = int(request.args.get("limit") or 200)
    except Exception:
        limit = 200
    try:
        return jsonify(
            {
                "items": get_app_services().receipt_db.list_review_queue(
                    status=status_filter,
                    limit=limit,
                )
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@review_bp.post("/api/review-queue/<job_id>/status")
def review_queue_update_status(job_id: str):
    payload = request.get_json(silent=True) or {}
    status_value = str(payload.get("status") or "").strip()
    if not status_value:
        return jsonify({"error": "missing status"}), 400

    services = get_app_services()
    try:
        result = services.receipt_db.update_review_status(job_id, status_value)
        services.job_store.update(job_id, review_queue_status=status_value)
        services.job_store.add_event(
            job_id,
            {
                "stage": "review_queue",
                "status": "done",
                "message": f"Review queue status changed to {status_value}.",
                "details": result,
            },
        )
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
