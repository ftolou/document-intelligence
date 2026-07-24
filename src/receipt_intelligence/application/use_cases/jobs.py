"""Receipt job submission, lookup, and artifact use cases."""

from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from receipt_intelligence.application.errors import (
    InvalidRequestError,
    ResourceNotFoundError,
    UnsupportedResourceError,
)
from receipt_intelligence.application.ports.jobs import JobProcessor, JobRepository
from receipt_intelligence.utils.filenames import safe_filename


@dataclass(frozen=True, slots=True)
class SubmitReceiptCommand:
    filename: str
    stream: BinaryIO
    options: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StartBatchCommand:
    folder_path: str | None
    recursive: bool
    max_files: int
    options: dict[str, Any]


class JobUseCases:
    """Application boundary for receipt processing jobs.

    Thread-backed execution remains intentionally encapsulated here until the
    durable dispatcher migration. HTTP routes never create or manage workers.
    """

    def __init__(self, store: JobRepository, processor: JobProcessor) -> None:
        self._store = store
        self._processor = processor

    def submit_receipt(self, command: SubmitReceiptCommand) -> dict[str, Any]:
        filename = str(command.filename or "").strip()
        if not filename:
            raise InvalidRequestError("Missing file field named 'file'.")
        if not self._processor.allowed_file(filename):
            raise UnsupportedResourceError("Unsupported receipt image type.")

        job_id = uuid.uuid4().hex[:12]
        job_dir = self._store.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        safe_name = safe_filename(filename, fallback=f"receipt_{job_id}.jpg")
        image_path = job_dir / safe_name
        with image_path.open("wb") as output:
            shutil.copyfileobj(command.stream, output)

        self._store.create(
            job_id,
            {
                "filename": safe_name,
                "image_path": str(image_path),
                "options": command.options,
            },
        )
        threading.Thread(
            target=self._processor.run_job,
            args=(job_id, image_path, command.options),
            daemon=True,
        ).start()
        return {"job_id": job_id, "state": "queued"}

    def start_batch(self, command: StartBatchCommand) -> dict[str, Any]:
        try:
            folder = self._processor.resolve_batch_folder(command.folder_path)
            image_paths = self._processor.list_batch_images(
                folder,
                recursive=command.recursive,
                max_files=command.max_files,
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        if not image_paths:
            raise InvalidRequestError(f"No supported receipt images found in {folder}.")

        batch_id = "batch_" + uuid.uuid4().hex[:10]
        batch_dir = self._store.job_dir(batch_id)
        batch_dir.mkdir(parents=True, exist_ok=True)
        self._store.create(
            batch_id,
            {
                "type": "batch",
                "folder_path": str(folder),
                "recursive": command.recursive,
                "max_files": command.max_files,
                "total": len(image_paths),
                "completed": 0,
                "failed": 0,
                "items": [],
                "options": command.options,
            },
        )
        threading.Thread(
            target=self._processor.run_batch_job,
            args=(batch_id, image_paths, command.options),
            daemon=True,
        ).start()
        return {
            "batch_id": batch_id,
            "job_id": batch_id,
            "state": "queued",
            "total": len(image_paths),
        }

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self._store.get(job_id)
        if job is None:
            raise ResourceNotFoundError("job not found")
        return job

    def list_jobs(self, *, limit: int = 25) -> list[dict[str, Any]]:
        return self._store.list_recent(limit=limit)

    def get_manifest(self, job_id: str) -> dict[str, Any]:
        self.get_job(job_id)
        manifest = self._store.get_manifest(job_id, create_if_missing=True)
        if manifest is None:
            raise ResourceNotFoundError("manifest not found")
        return manifest

    def get_artifact(self, job_id: str, filename: str) -> Path:
        job_dir = self._store.job_dir(job_id).resolve()
        requested = (job_dir / Path(filename).name).resolve()
        try:
            requested.relative_to(job_dir)
        except ValueError as exc:
            raise InvalidRequestError("invalid artifact path") from exc
        if not requested.exists() or not requested.is_file():
            raise ResourceNotFoundError("artifact not found")
        return requested


__all__ = ["JobUseCases", "StartBatchCommand", "SubmitReceiptCommand"]
