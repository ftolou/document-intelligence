"""Ports required by receipt job submission and background execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

JobKind = Literal["receipt", "batch"]


@dataclass(frozen=True, slots=True)
class JobDispatchRequest:
    """Serializable description of work accepted by a background dispatcher."""

    job_id: str
    kind: JobKind
    options: dict[str, Any]
    image_path: Path | None = None
    image_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("job_id must not be empty")
        if self.kind == "receipt":
            if self.image_path is None or self.image_paths:
                raise ValueError("receipt jobs require image_path only")
        elif self.kind == "batch":
            if self.image_path is not None or not self.image_paths:
                raise ValueError("batch jobs require one or more image_paths")
        else:  # pragma: no cover - Literal protects typed callers.
            raise ValueError(f"Unsupported job kind: {self.kind}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "image_path": str(self.image_path) if self.image_path is not None else None,
            "image_paths": [str(path) for path in self.image_paths],
            "options": dict(self.options),
        }

    @classmethod
    def from_payload(cls, job_id: str, payload: dict[str, Any]) -> JobDispatchRequest:
        kind = str(payload.get("kind") or "").strip()
        if kind == "receipt":
            image_path = payload.get("image_path")
            if not image_path:
                raise ValueError(f"Receipt job {job_id} has no image_path")
            return cls(
                job_id=job_id,
                kind="receipt",
                image_path=Path(str(image_path)),
                options=dict(payload.get("options") or {}),
            )
        if kind == "batch":
            image_paths = tuple(Path(str(value)) for value in payload.get("image_paths") or [])
            return cls(
                job_id=job_id,
                kind="batch",
                image_paths=image_paths,
                options=dict(payload.get("options") or {}),
            )
        raise ValueError(f"Job {job_id} has unsupported dispatch kind: {kind!r}")


class JobQueueFullError(RuntimeError):
    """Raised when the bounded local worker queue cannot accept more work."""


class JobRepository(Protocol):
    def job_dir(self, job_id: str) -> Path: ...

    def create(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    def get(self, job_id: str) -> dict[str, Any] | None: ...

    def list_recent(self, limit: int = 25) -> list[dict[str, Any]]: ...

    def list_dispatchable(self) -> list[dict[str, Any]]: ...

    def get_manifest(
        self,
        job_id: str,
        *,
        create_if_missing: bool = True,
    ) -> dict[str, Any] | None: ...

    def update(self, job_id: str, **fields: Any) -> None: ...

    def begin_attempt(self, job_id: str) -> None: ...

    def complete(self, job_id: str) -> None: ...

    def fail(self, job_id: str, error: dict[str, Any]) -> None: ...

    def try_claim(self, job_id: str, owner_id: str, *, lease_seconds: float) -> bool: ...

    def renew_claim(
        self,
        job_id: str,
        owner_id: str,
        *,
        lease_seconds: float,
    ) -> bool: ...

    def release_claim(self, job_id: str, owner_id: str) -> None: ...

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


class JobDispatcher(Protocol):
    def submit(self, request: JobDispatchRequest) -> None: ...

    def recover_pending(self) -> int: ...

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None: ...


__all__ = [
    "JobDispatchRequest",
    "JobDispatcher",
    "JobKind",
    "JobProcessor",
    "JobQueueFullError",
    "JobRepository",
]
