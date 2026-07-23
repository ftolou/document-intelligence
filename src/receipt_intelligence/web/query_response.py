"""Normalize the single RAG-SQL HTTP response contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ERROR_MESSAGES = {
    "missing_question": "Enter a question about your approved receipts.",
    "invalid_request": "The receipt query request is invalid.",
    "unsupported_request_field": "The request contains an unsupported field.",
    "embedding_unavailable": "Semantic receipt search is currently unavailable.",
    "question_analysis_failed": "The question could not be interpreted reliably.",
    "semantic_retrieval_failed": "Approved receipt products could not be searched.",
    "candidate_resolution_failed": "Matching receipt products could not be resolved.",
    "sql_planning_failed": "A database query could not be created for this question.",
    "sql_validation_failed": "The generated query did not pass the safety checks.",
    "sql_execution_failed": "The validated query could not be executed.",
    "result_formatting_failed": "The query result could not be formatted safely.",
    "query_execution_failed": "The receipt query could not be completed.",
}


def user_message_for_error(error_code: str | None) -> str:
    return _ERROR_MESSAGES.get(
        str(error_code or "").strip(),
        "The receipt query could not be completed.",
    )


def _normalized_status(payload: Mapping[str, Any]) -> str:
    status = payload.get("status")
    if not status and isinstance(payload.get("execution"), Mapping):
        status = payload["execution"].get("status")
    value = str(status or ("error" if payload.get("error") else "completed")).strip().lower()
    if value in {"done", "success", "ok"}:
        return "completed"
    return value or "completed"


def normalize_query_response(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable RAG-SQL response envelope."""

    payload = dict(result)
    payload["strategy"] = "rag_sql"
    payload["status"] = _normalized_status(payload)
    payload.setdefault("answer", None)
    payload.setdefault("data", None)
    payload.setdefault("clarification_question", None)
    payload.setdefault("error_code", None)
    payload.setdefault("error", None)
    return payload


def query_error_payload(*, error_code: str, error: str) -> dict[str, Any]:
    return {
        "strategy": "rag_sql",
        "status": "error",
        "answer": user_message_for_error(error_code),
        "data": None,
        "clarification_question": None,
        "error_code": error_code,
        "error": error,
    }


__all__ = ["normalize_query_response", "query_error_payload", "user_message_for_error"]
