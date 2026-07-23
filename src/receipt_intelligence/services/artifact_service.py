"""Artifact URL and path helpers shared by web and background jobs."""

from __future__ import annotations

from pathlib import Path

from receipt_intelligence.storage.job_store import JobStore


def artifact_url(job_id: str, path: Path) -> str:
    return f"/api/artifact/{job_id}/{path.name}"


def safe_artifact_path(store: JobStore, job_id: str, filename: str) -> Path | None:
    job_dir = store.job_dir(job_id).resolve()
    requested = (job_dir / filename).resolve()
    try:
        requested.relative_to(job_dir)
    except ValueError:
        return None
    return requested
