"""JSON-backed job store for the canonical ``var/jobs`` runtime root."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from receipt_intelligence.runtime.manifest import JobManifestStore

_DISPATCHABLE_STATES = {"queued", "running"}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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

    def claim_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / ".worker_claim.json"

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
        now = _utc_now_iso()
        job = {
            "job_id": job_id,
            "state": "queued",
            "created_at": now,
            "queued_at": now,
            "started_at": None,
            "finished_at": None,
            "updated_at": now,
            "attempt_count": 0,
            "events": [],
            "result": None,
            "error": None,
            **payload,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._write(job_id)
            self._update_manifest(job)
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
        return dict(job)

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
            job = self._load_for_update(job_id)
            job.update(fields)
            job["updated_at"] = _utc_now_iso()
            self._write(job_id)
            self._update_manifest(job)

    def begin_attempt(self, job_id: str) -> None:
        with self._lock:
            job = self._load_for_update(job_id)
            now = _utc_now_iso()
            job.update(
                state="running",
                started_at=now,
                finished_at=None,
                updated_at=now,
                attempt_count=int(job.get("attempt_count") or 0) + 1,
                error=None,
            )
            self._write(job_id)
            self._update_manifest(job)

    def complete(self, job_id: str) -> None:
        with self._lock:
            job = self._load_for_update(job_id)
            now = _utc_now_iso()
            job.update(state="completed", finished_at=now, updated_at=now, error=None)
            self._write(job_id)
            self._update_manifest(job)

    def fail(self, job_id: str, error: dict[str, Any]) -> None:
        with self._lock:
            job = self._load_for_update(job_id)
            now = _utc_now_iso()
            job.update(state="failed", finished_at=now, updated_at=now, error=dict(error))
            self._write(job_id)
            self._update_manifest(job)

    def add_event(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            job = self._load_for_update(job_id)
            event = dict(event)
            event.setdefault("time", time.strftime("%H:%M:%S"))
            job.setdefault("events", []).append(event)
            job["updated_at"] = _utc_now_iso()
            stage = event.get("stage")
            status = event.get("status")
            if stage:
                job["current_stage"] = stage
            if status:
                job["current_status"] = status
            self._write(job_id)
            self._update_manifest(
                job,
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

    def list_dispatchable(self) -> list[dict[str, Any]]:
        jobs = self.list_recent(limit=1_000_000)
        return [
            job
            for job in reversed(jobs)
            if str(job.get("state") or "") in _DISPATCHABLE_STATES
            and isinstance(job.get("dispatch"), dict)
        ]

    def try_claim(self, job_id: str, owner_id: str, *, lease_seconds: float) -> bool:
        path = self.claim_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lease_seconds = max(1.0, float(lease_seconds))

        for _ in range(3):
            now = time.time()
            payload = {
                "job_id": job_id,
                "owner_id": owner_id,
                "claimed_at": _utc_now_iso(),
                "expires_at_epoch": now + lease_seconds,
            }
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if float(existing.get("expires_at_epoch") or 0.0) > now:
                        return False
                except Exception:
                    pass
                stale = path.with_name(f"{path.name}.expired.{uuid.uuid4().hex}")
                try:
                    path.replace(stale)
                except FileNotFoundError:
                    continue
                try:
                    stale.unlink(missing_ok=True)
                except OSError:
                    pass
                continue

            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            return True
        return False

    def renew_claim(
        self,
        job_id: str,
        owner_id: str,
        *,
        lease_seconds: float,
    ) -> bool:
        path = self.claim_path(job_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if payload.get("owner_id") != owner_id:
            return False
        payload["expires_at_epoch"] = time.time() + max(1.0, float(lease_seconds))
        payload["renewed_at"] = _utc_now_iso()
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError:
            temporary.unlink(missing_ok=True)
            return False
        return True

    def release_claim(self, job_id: str, owner_id: str) -> None:
        path = self.claim_path(job_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            payload = {}
        if payload.get("owner_id") not in {None, owner_id}:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _load_for_update(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        loaded = self.get(job_id)
        if loaded is not None:
            return loaded
        job = {"job_id": job_id, "events": []}
        self._jobs[job_id] = job
        return job

    def _update_manifest(
        self,
        job: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        job_id = str(job["job_id"])
        merged_metadata = {
            "filename": job.get("filename"),
            "batch_id": job.get("batch_id"),
            "attempt_count": job.get("attempt_count"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
        }
        if metadata:
            merged_metadata.update(metadata)
        self.manifests.update(
            self.job_dir(job_id),
            job_id,
            state=str(job.get("state") or "unknown"),
            job_type=str(job.get("type") or "single"),
            legacy_layout=False,
            metadata=merged_metadata,
        )

    def _write(self, job_id: str) -> None:
        path = self.status_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._jobs[job_id], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
