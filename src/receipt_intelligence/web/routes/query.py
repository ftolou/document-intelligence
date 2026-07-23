"""Ask Your Receipts API backed exclusively by RAG-SQL LangGraph."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from receipt_intelligence.web.dependencies import get_app_services
from receipt_intelligence.web.query_response import normalize_query_response, query_error_payload

query_bp = Blueprint("query", __name__)
_ALLOWED_FIELDS = {"question", "limit"}


@query_bp.post("/api/ask-receipts")
def ask_receipts():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(
            query_error_payload(error_code="invalid_request", error="Expected JSON object.")
        ), 400

    unsupported_fields = sorted(set(payload) - _ALLOWED_FIELDS)
    if unsupported_fields:
        return (
            jsonify(
                query_error_payload(
                    error_code="unsupported_request_field",
                    error=f"Unsupported request field(s): {', '.join(unsupported_fields)}.",
                )
            ),
            400,
        )

    question = str(payload.get("question") or "").strip()
    if not question:
        return (
            jsonify(query_error_payload(error_code="missing_question", error="Missing question.")),
            400,
        )
    try:
        limit = max(1, min(100, int(payload.get("limit", 25))))
    except (TypeError, ValueError):
        limit = 25

    services = get_app_services()
    try:
        result = services.receipt_query_service.execute(question, limit=limit)
        return jsonify(normalize_query_response(result))
    except Exception as exc:
        return (
            jsonify(
                query_error_payload(
                    error_code="query_execution_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            ),
            500,
        )
