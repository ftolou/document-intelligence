"""Receipt database management endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from receipt_intelligence.application.errors import ApplicationError
from receipt_intelligence.web.dependencies import get_app_services
from receipt_intelligence.web.errors import application_error_response, unexpected_error_response
from receipt_intelligence.web.presentation import present_resources, present_review
from receipt_intelligence.web.request_parsing import as_bool

receipts_bp = Blueprint("receipts", __name__)


def _review_input() -> tuple[dict, list, dict]:
    payload = request.get_json(silent=True) or {}
    fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    return fields, items, review


@receipts_bp.get("/api/receipt-db/summary")
def receipt_db_summary():
    return jsonify(get_app_services().receipts.summary())


@receipts_bp.post("/api/receipts/import/<job_id>")
def import_receipt_job(job_id: str):
    try:
        return jsonify(present_resources(get_app_services().receipts.import_job(job_id)))
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)


@receipts_bp.get("/api/receipt-db/receipts")
def receipt_db_receipts():
    try:
        limit = int(request.args.get("limit") or 200)
    except (TypeError, ValueError):
        limit = 200
    try:
        return jsonify(
            {"receipts": get_app_services().receipts.list_receipts(limit=limit)}
        )
    except Exception as exc:
        return unexpected_error_response(exc)


@receipts_bp.get("/api/receipt-db/receipts/<int:receipt_id>")
def receipt_db_get_receipt(receipt_id: int):
    try:
        return jsonify({"receipt": get_app_services().receipts.get_receipt(receipt_id)})
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)


@receipts_bp.get("/api/receipt-db/receipts/<int:receipt_id>/review")
def receipt_db_get_review(receipt_id: int):
    try:
        return jsonify(present_review(get_app_services().receipts.load_review(receipt_id)))
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)


@receipts_bp.put("/api/receipt-db/receipts/<int:receipt_id>/review")
def receipt_db_update_review(receipt_id: int):
    fields, item_corrections, review = _review_input()
    try:
        result = get_app_services().receipts.save_review(
            receipt_id,
            fields=fields,
            item_corrections=item_corrections,
            review=review,
        )
        return jsonify(present_review(result))
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)


@receipts_bp.delete("/api/receipt-db/receipts/<receipt_id>")
def receipt_db_delete_receipt(receipt_id: str):
    try:
        return jsonify(get_app_services().receipts.delete_receipt(receipt_id))
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)


@receipts_bp.post("/api/receipt-db/delete-all")
def receipt_db_delete_all():
    payload = request.get_json(silent=True) or {}
    try:
        result = get_app_services().receipts.delete_all(
            confirmation=str(payload.get("confirm") or ""),
            include_review_queue=as_bool(payload.get("include_review_queue"), False),
        )
        return jsonify(result)
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)
