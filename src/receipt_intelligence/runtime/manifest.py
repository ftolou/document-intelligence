"""Per-job manifest persistence for discoverable runtime artifacts."""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "job_manifest_v1"


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def infer_artifact_category(key: str, path: Path) -> str:
    text = f"{key} {path.name}".lower()
    if "image" in text or path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tif",
        ".tiff",
        ".bmp",
    }:
        return "input"
    if "ocr" in text:
        return "ocr"
    if "prompt" in text or "raw" in text or "llm" in text:
        return "llm"
    if "metric" in text or "trace" in text or "telemetry" in text:
        return "observability"
    if "validation" in text or "report" in text:
        return "validation"
    if "review" in text or "approved" in text:
        return "review"
    if "receipt" in text or "final" in text or "categor" in text:
        return "final"
    if "batch" in text:
        return "batch"
    return "other"


class JobManifestStore:
    """Write and update ``manifest.json`` files inside job directories."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @staticmethod
    def path(job_dir: Path) -> Path:
        return Path(job_dir) / "manifest.json"

    def load(self, job_dir: Path) -> dict[str, Any] | None:
        path = self.path(job_dir)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def ensure(
        self,
        job_dir: Path,
        job_id: str,
        *,
        job_type: str = "single",
        legacy_layout: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            directory = Path(job_dir)
            directory.mkdir(parents=True, exist_ok=True)
            current = self.load(directory)
            if current is not None:
                return current
            now = _timestamp()
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "job_id": job_id,
                "job_type": job_type,
                "state": "unknown",
                "created_at": now,
                "updated_at": now,
                "legacy_layout": bool(legacy_layout),
                "artifacts": {},
                "metadata": dict(metadata or {}),
            }
            self._write(directory, manifest)
            return manifest

    def update(
        self,
        job_dir: Path,
        job_id: str,
        *,
        state: str | None = None,
        metadata: dict[str, Any] | None = None,
        job_type: str | None = None,
        legacy_layout: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            manifest = self.ensure(
                job_dir,
                job_id,
                job_type=job_type or "single",
                legacy_layout=legacy_layout,
            )
            if state is not None:
                manifest["state"] = state
            if job_type is not None:
                manifest["job_type"] = job_type
            if metadata:
                manifest.setdefault("metadata", {}).update(metadata)
            manifest["updated_at"] = _timestamp()
            self._write(Path(job_dir), manifest)
            return manifest

    def register_artifact(
        self,
        job_dir: Path,
        job_id: str,
        key: str,
        path: Path,
        *,
        category: str | None = None,
        media_type: str | None = None,
        legacy_layout: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            directory = Path(job_dir).resolve()
            artifact_path = Path(path).resolve()
            try:
                relative = artifact_path.relative_to(directory)
            except ValueError as exc:
                raise ValueError(
                    f"Artifact {artifact_path} is outside job directory {directory}"
                ) from exc
            manifest = self.ensure(
                directory,
                job_id,
                legacy_layout=legacy_layout,
            )
            entry: dict[str, Any] = {
                "path": relative.as_posix(),
                "category": category or infer_artifact_category(key, artifact_path),
                "exists": artifact_path.exists(),
                "media_type": media_type
                or mimetypes.guess_type(artifact_path.name)[0]
                or "application/octet-stream",
                "updated_at": _timestamp(),
            }
            if artifact_path.exists() and artifact_path.is_file():
                try:
                    entry["size_bytes"] = artifact_path.stat().st_size
                except OSError:
                    pass
            if metadata:
                entry["metadata"] = dict(metadata)
            manifest.setdefault("artifacts", {})[key] = entry
            manifest["updated_at"] = _timestamp()
            self._write(directory, manifest)
            return entry

    def scan_existing(
        self,
        job_dir: Path,
        job_id: str,
        *,
        legacy_layout: bool = False,
    ) -> dict[str, Any]:
        directory = Path(job_dir)
        status_path = directory / "job_status.json"
        status: dict[str, Any] = {}
        if status_path.exists():
            try:
                loaded = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    status = loaded
            except Exception:
                status = {}
        manifest = self.update(
            directory,
            job_id,
            state=str(status.get("state") or "unknown"),
            job_type=str(status.get("type") or "single"),
            legacy_layout=legacy_layout,
            metadata={
                "filename": status.get("filename"),
                "batch_id": status.get("batch_id"),
            },
        )
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            relative_key = path.relative_to(directory).as_posix()
            key = relative_key.replace("/", "__")
            self.register_artifact(
                directory,
                job_id,
                key,
                path,
                legacy_layout=legacy_layout,
            )
        return self.load(directory) or manifest

    def _write(self, job_dir: Path, manifest: dict[str, Any]) -> None:
        path = self.path(job_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
