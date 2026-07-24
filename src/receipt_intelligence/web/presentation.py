"""HTTP presentation helpers for transport-neutral application results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from receipt_intelligence.application.resources import (
    artifact_reference_parts,
    is_artifact_reference,
)


def present_resources(value: Any) -> Any:
    """Recursively convert application resource references into HTTP URLs."""

    if is_artifact_reference(value):
        job_id, filename = artifact_reference_parts(value)
        return f"/api/artifact/{quote(job_id, safe='')}/{quote(filename, safe='')}"
    if isinstance(value, Mapping):
        return {str(key): present_resources(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [present_resources(item) for item in value]
    return value


def present_job_submission(payload: Mapping[str, Any]) -> dict[str, Any]:
    response = dict(present_resources(payload))
    job_id = str(response.get("job_id") or "")
    if job_id:
        response["status_url"] = f"/api/status/{quote(job_id, safe='')}"
    return response


def present_review(payload: Mapping[str, Any]) -> dict[str, Any]:
    response = dict(present_resources(payload))
    receipt_id = response.get("receipt_db_id") or response.get("receipt_id")
    job_id = str(response.get("job_id") or "").strip()
    if response.get("source") == "database" and receipt_id is not None:
        response["save_url"] = f"/api/receipt-db/receipts/{int(receipt_id)}/review"
        response["save_method"] = "PUT"
    elif job_id:
        response["save_url"] = f"/api/review/{quote(job_id, safe='')}"
        response["save_method"] = "POST"
    return response


__all__ = ["present_job_submission", "present_resources", "present_review"]
