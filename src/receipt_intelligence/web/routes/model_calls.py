"""HTTP endpoints for model-call usage and estimated costs."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from receipt_intelligence.web.dependencies import get_app_services

model_calls_bp = Blueprint("model_calls", __name__)


def _hours() -> int | None:
    raw = request.args.get("hours", "24").strip().lower()
    if raw in {"all", "0", ""}:
        return None
    return max(1, min(24 * 365, int(raw)))


def _filters() -> dict[str, str | None]:
    names = ("provider", "model", "operation", "status")
    return {name: request.args.get(name) or None for name in names}


@model_calls_bp.get("/api/model-calls/summary")
def model_call_summary():
    return jsonify(get_app_services().model_calls.summary(hours=_hours(), **_filters()))


@model_calls_bp.get("/api/model-calls")
def model_call_list():
    limit = max(1, min(500, int(request.args.get("limit", "100"))))
    offset = max(0, int(request.args.get("offset", "0")))
    calls = get_app_services().model_calls.list_calls(
        hours=_hours(),
        limit=limit,
        offset=offset,
        **_filters(),
    )
    return jsonify({"calls": calls})


@model_calls_bp.get("/api/model-pricing")
def model_pricing_list():
    use_cases = get_app_services().model_calls
    return jsonify({"pricing": use_cases.pricing(), "models": use_cases.models()})


@model_calls_bp.put("/api/model-pricing")
def model_pricing_save():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object."}), 400
    try:
        return jsonify(get_app_services().model_calls.save_pricing(payload))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


__all__ = ["model_calls_bp"]
