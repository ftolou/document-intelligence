"""Receipt database management endpoints."""

from __future__ import annotations

import json

from flask import Blueprint, jsonify, request

from receipt_intelligence.services.database_receipt_editor import DatabaseReceiptEditor
from receipt_intelligence.services.review_service import ReviewService
from receipt_intelligence.web.dependencies import get_app_services
from receipt_intelligence.web.request_parsing import as_bool

receipts_bp = Blueprint("receipts", __name__)


@receipts_bp.get("/api/receipt-db/summary")
def receipt_db_summary():
    return jsonify(get_app_services().receipt_db.summary())


@receipts_bp.post("/api/receipts/import/<job_id>")
def import_receipt_job(job_id: str):
    services = get_app_services()
    store = services.job_store
    receipt_db = services.receipt_db
    review_service = ReviewService(store, receipt_db)

    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404

    approved_path = review_service.approved_receipt_path(job_id)
    source_path = (
        approved_path if approved_path.exists() else review_service.final_receipt_path(job_id)
    )
    if source_path is None or not source_path.exists():
        return jsonify({"error": "no approved/final receipt JSON found for this job"}), 404

    try:
        receipt = json.loads(source_path.read_text(encoding="utf-8"))
        db_import = review_service.import_reviewed_receipt(
            job_id,
            receipt,
            source_path,
            source_path,
        )
        receipt_db.update_review_status(
            job_id,
            "imported",
            receipt_db_id=db_import.get("receipt_db_id"),
        )
        store.update(job_id, receipt_db_import=db_import)
        store.add_event(
            job_id,
            {
                "stage": "receipt_db",
                "status": "done",
                "message": (
                    f"Receipt imported into local DB with {db_import['item_count']} item(s)."
                ),
                "details": db_import,
            },
        )
        return jsonify(
            {
                "ok": True,
                "job_id": job_id,
                "receipt_db_import": db_import,
                "summary": receipt_db.summary(),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@receipts_bp.get("/api/receipt-db/receipts")
def receipt_db_receipts():
    try:
        limit = int(request.args.get("limit") or 200)
    except Exception:
        limit = 200
    try:
        return jsonify({"receipts": get_app_services().receipt_db.list_receipts(limit=limit)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@receipts_bp.get("/api/receipt-db/receipts/<int:receipt_id>")
def receipt_db_get_receipt(receipt_id: int):
    """Return safe metadata required to open an existing receipt review."""

    try:
        receipt = get_app_services().receipt_db.get_receipt(receipt_id)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    if receipt is None:
        return jsonify({"error": "receipt not found"}), 404
    return jsonify({"receipt": receipt})


@receipts_bp.get("/api/receipt-db/receipts/<int:receipt_id>/review")
def receipt_db_get_review(receipt_id: int):
    """Load an editable receipt directly from authoritative SQLite rows."""

    services = get_app_services()
    editor = DatabaseReceiptEditor(
        services.receipt_db,
        ReviewService(services.job_store, services.receipt_db),
    )
    try:
        return jsonify(editor.load(receipt_id))
    except KeyError:
        return jsonify({"error": "receipt not found"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@receipts_bp.put("/api/receipt-db/receipts/<int:receipt_id>/review")
def receipt_db_update_review(receipt_id: int):
    """Update a reviewed receipt by database ID and selectively refresh embeddings."""

    payload = request.get_json(silent=True) or {}
    fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
    item_corrections = payload.get("items") if isinstance(payload.get("items"), list) else []
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    services = get_app_services()
    editor = DatabaseReceiptEditor(
        services.receipt_db,
        ReviewService(services.job_store, services.receipt_db),
    )
    try:
        return jsonify(
            editor.save(
                receipt_id,
                fields=fields,
                item_corrections=item_corrections,
                review=review,
            )
        )
    except KeyError:
        return jsonify({"error": "receipt not found"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@receipts_bp.delete("/api/receipt-db/receipts/<receipt_id>")
def receipt_db_delete_receipt(receipt_id: str):
    receipt_db = get_app_services().receipt_db
    try:
        if receipt_id.startswith("job:"):
            result = receipt_db.delete_receipt(job_id=receipt_id.split(":", 1)[1])
        else:
            result = receipt_db.delete_receipt(receipt_id=int(receipt_id))
        return jsonify({"ok": True, "result": result, "summary": receipt_db.summary()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@receipts_bp.post("/api/receipt-db/delete-all")
def receipt_db_delete_all():
    payload = request.get_json(silent=True) or {}
    if payload.get("confirm") != "DELETE_ALL_RECEIPTS":
        return jsonify({"error": "confirmation required: DELETE_ALL_RECEIPTS"}), 400
    include_review_queue = as_bool(payload.get("include_review_queue"), False)
    receipt_db = get_app_services().receipt_db
    try:
        result = receipt_db.delete_all_receipt_data(include_review_queue=include_review_queue)
        return jsonify({"ok": True, "deleted": result, "summary": receipt_db.summary()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
