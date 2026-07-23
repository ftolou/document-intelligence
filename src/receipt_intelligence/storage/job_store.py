"""JSON-backed job store for the canonical ``var/jobs`` runtime root."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from receipt_intelligence.runtime.manifest import JobManifestStore


class JobStore:
    def __init__(
        self,
        results_dir: Path,
        *,
        manifest_store: JobManifestStore | None = None,
    ) -> None:
        self.results_dir = Path(results_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.manifests = manifest_store or JobManifestStore()
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}

    @property
    def read_roots(self) -> tuple[Path, ...]:
        return (self.results_dir,)

    def job_dir(self, job_id: str) -> Path:
        return self.results_dir / job_id

    def status_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job_status.json"

    def manifest_path(self, job_id: str) -> Path:
        return self.manifests.path(self.job_dir(job_id))

    def get_manifest(self, job_id: str, *, create_if_missing: bool = True) -> dict[str, Any] | None:
        directory = self.job_dir(job_id)
        manifest = self.manifests.load(directory)
        if manifest is None and create_if_missing and directory.exists():
            manifest = self.manifests.scan_existing(directory, job_id, legacy_layout=False)
        return manifest

    def register_artifact(
        self,
        job_id: str,
        key: str,
        path: Path,
        *,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.manifests.register_artifact(
            self.job_dir(job_id),
            job_id,
            key,
            Path(path),
            category=category,
            metadata=metadata,
            legacy_layout=False,
        )

    def create(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        job = {
            "job_id": job_id,
            "state": "queued",
            "created_at": now,
            "updated_at": now,
            "events": [],
            "result": None,
            "error": None,
            **payload,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._write(job_id)
            self.manifests.update(
                self.job_dir(job_id),
                job_id,
                state=str(job.get("state") or "queued"),
                job_type=str(job.get("type") or "single"),
                legacy_layout=False,
                metadata={
                    "filename": job.get("filename"),
                    "batch_id": job.get("batch_id"),
                },
            )
            self.register_artifact(
                job_id,
                "job_status",
                self.status_path(job_id),
                category="state",
            )
            image_path = job.get("image_path")
            if image_path:
                path = Path(str(image_path))
                if path.exists():
                    self.register_artifact(job_id, "input_image", path, category="input")
        return job

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]
            path = self.status_path(job_id)
            if path.exists():
                try:
                    job = json.loads(path.read_text(encoding="utf-8"))
                    self._jobs[job_id] = job
                    return job
                except Exception:
                    return None
        return None

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.setdefault(job_id, {"job_id": job_id, "events": []})
            job.update(fields)
            job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._write(job_id)
            self.manifests.update(
                self.job_dir(job_id),
                job_id,
                state=str(job.get("state") or "unknown"),
                job_type=str(job.get("type") or "single"),
                legacy_layout=False,
            )

    def add_event(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.setdefault(job_id, {"job_id": job_id, "events": []})
            event = dict(event)
            event.setdefault("time", time.strftime("%H:%M:%S"))
            job.setdefault("events", []).append(event)
            job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            stage = event.get("stage")
            status = event.get("status")
            if stage:
                job["current_stage"] = stage
            if status:
                job["current_status"] = status
            self._write(job_id)
            self.manifests.update(
                self.job_dir(job_id),
                job_id,
                state=str(job.get("state") or status or "unknown"),
                job_type=str(job.get("type") or "single"),
                legacy_layout=False,
                metadata={"current_stage": stage, "current_status": status},
            )

    def list_recent(self, limit: int = 25) -> list[dict[str, Any]]:
        if not self.results_dir.exists():
            return []

        candidates = sorted(
            self.results_dir.glob("*/job_status.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        jobs: list[dict[str, Any]] = []
        for path in candidates:
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if len(jobs) >= limit:
                break
        return jobs

    def _write(self, job_id: str) -> None:
        path = self.status_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._jobs[job_id], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
