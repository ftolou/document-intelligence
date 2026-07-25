"""Ask Your Receipts API backed exclusively by the application query use case."""

from __future__ import annotations

from collections.abc import Mapping

from flask import Blueprint, jsonify, request

from receipt_intelligence.application.errors import InvalidRequestError
from receipt_intelligence.web.dependencies import get_app_services
from receipt_intelligence.web.query_response import normalize_query_response, query_error_payload

query_bp = Blueprint("query", __name__)


@query_bp.post("/api/ask-receipts")
def ask_receipts():
    try:
        result = get_app_services().ask_receipts.execute(request.get_json(silent=True))
        return jsonify(normalize_query_response(result))
    except InvalidRequestError as exc:
        payload = query_error_payload(error_code=exc.code, error=str(exc))
        _copy_diagnostic_log(payload, exc)
        return jsonify(payload), 400
    except Exception as exc:
        payload = query_error_payload(
            error_code="query_execution_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        _copy_diagnostic_log(payload, exc)
        return jsonify(payload), 500


def _copy_diagnostic_log(payload: dict[str, object], exc: Exception) -> None:
    metadata = getattr(exc, "diagnostic_log", None)
    if isinstance(metadata, Mapping):
        payload["diagnostic_log"] = dict(metadata)
