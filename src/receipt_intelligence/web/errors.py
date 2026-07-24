"""Mapping from application errors to Flask responses."""

from __future__ import annotations

from flask import jsonify

from receipt_intelligence.application.errors import (
    ApplicationError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)


def application_error_response(error: ApplicationError):
    if isinstance(error, ResourceNotFoundError):
        status = 404
    elif isinstance(error, ServiceUnavailableError):
        status = 503
    else:
        status = 400
    payload = {"error": str(error)}
    if error.code not in {"invalid_request", "not_found", "application_error"}:
        payload["error_code"] = error.code
    return jsonify(payload), status


def unexpected_error_response(error: Exception):
    return jsonify({"error": str(error)}), 500


__all__ = ["application_error_response", "unexpected_error_response"]
