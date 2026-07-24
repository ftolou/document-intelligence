"""Ports required by receipt job use cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class JobRepository(Protocol):
    def job_dir(self, job_id: str) -> Path: ...

    def create(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    def get(self, job_id: str) -> dict[str, Any] | None: ...

    def list_recent(self, limit: int = 25) -> list[dict[str, Any]]: ...

    def get_manifest(
        self,
        job_id: str,
        *,
        create_if_missing: bool = True,
    ) -> dict[str, Any] | None: ...

    def update(self, job_id: str, **fields: Any) -> None: ...

    def add_event(self, job_id: str, event: dict[str, Any]) -> None: ...

    def register_artifact(
        self,
        job_id: str,
        key: str,
        path: Path,
        *,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class JobProcessor(Protocol):
    def allowed_file(self, filename: str) -> bool: ...

    def run_job(self, job_id: str, image_path: Path, options: dict[str, Any]) -> None: ...

    def resolve_batch_folder(self, folder_text: str | None) -> Path: ...

    def list_batch_images(
        self,
        folder: Path,
        *,
        recursive: bool,
        max_files: int,
    ) -> list[Path]: ...

    def run_batch_job(
        self,
        batch_id: str,
        image_paths: list[Path],
        options: dict[str, Any],
    ) -> None: ...


__all__ = ["JobProcessor", "JobRepository"]
