"""Artifact reference and safe path helpers used by application services."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from receipt_intelligence.application.resources import artifact_reference
from receipt_intelligence.storage.job_store import JobStore


def artifact_resource(job_id: str, path: Path) -> dict[str, str]:
    """Return a transport-neutral reference for a job-owned artifact."""

    return artifact_reference(job_id, path)


def safe_artifact_path(store: JobStore, job_id: str, filename: str) -> Path | None:
    job_dir = store.job_dir(job_id).resolve()
    requested = (job_dir / Path(filename).name).resolve()
    try:
        requested.relative_to(job_dir)
    except ValueError:
        return None
    return requested


def artifact_path_from_reference(
    store: JobStore,
    value: Mapping[str, Any] | str | None,
    *,
    default_job_id: str | None = None,
) -> Path | None:
    """Resolve either a neutral artifact reference or a historical URL/path value."""

    if isinstance(value, Mapping) and value.get("resource_kind") == "job_artifact":
        job_id = str(value.get("job_id") or "").strip()
        filename = Path(str(value.get("filename") or "")).name
        if not job_id or not filename:
            return None
        return safe_artifact_path(store, job_id, filename)

    if not value or not default_job_id:
        return None
    text = str(value)
    filename = Path(text.split("?", 1)[0]).name
    if not filename:
        return None
    return safe_artifact_path(store, default_job_id, filename)


__all__ = ["artifact_path_from_reference", "artifact_resource", "safe_artifact_path"]
