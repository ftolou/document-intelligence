"""Human-review and review-queue endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from receipt_intelligence.application.errors import ApplicationError
from receipt_intelligence.web.dependencies import get_app_services
from receipt_intelligence.web.errors import application_error_response, unexpected_error_response
from receipt_intelligence.web.presentation import present_review

review_bp = Blueprint("review", __name__)


def _review_input() -> tuple[dict, list, dict, dict]:
    payload = request.get_json(silent=True) or {}
    fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    return fields, items, review, identity


@review_bp.get("/api/review/<job_id>")
def get_human_review(job_id: str):
    try:
        return jsonify(present_review(get_app_services().reviews.get_review(job_id)))
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)


@review_bp.post("/api/review/<job_id>")
def save_human_review(job_id: str):
    fields, item_corrections, review, identity = _review_input()
    try:
        result = get_app_services().reviews.save_review(
            job_id,
            fields=fields,
            item_corrections=item_corrections,
            review=review,
            identity=identity,
        )
        return jsonify(present_review(result))
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)


@review_bp.get("/api/review-queue")
def review_queue_list():
    status_filter = request.args.get("status") or "all"
    try:
        limit = int(request.args.get("limit") or 200)
    except (TypeError, ValueError):
        limit = 200
    try:
        return jsonify(
            {
                "items": get_app_services().reviews.list_queue(
                    status=status_filter,
                    limit=limit,
                )
            }
        )
    except Exception as exc:
        return unexpected_error_response(exc)


@review_bp.post("/api/review-queue/<job_id>/status")
def review_queue_update_status(job_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(
            get_app_services().reviews.update_queue_status(
                job_id,
                str(payload.get("status") or ""),
            )
        )
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)
