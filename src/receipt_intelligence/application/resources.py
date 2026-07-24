"""Transport-neutral resource references returned by application services."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

_RESOURCE_KIND = "job_artifact"


def artifact_reference(job_id: str, path: Path | str) -> dict[str, str]:
    """Return an opaque reference to an artifact owned by a job."""

    return {
        "resource_kind": _RESOURCE_KIND,
        "job_id": str(job_id),
        "filename": Path(path).name,
    }


def is_artifact_reference(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("resource_kind") == _RESOURCE_KIND
        and bool(value.get("job_id"))
        and bool(value.get("filename"))
    )


def artifact_reference_parts(value: Mapping[str, Any]) -> tuple[str, str]:
    if not is_artifact_reference(value):
        raise ValueError("value is not a job artifact reference")
    return str(value["job_id"]), Path(str(value["filename"])).name


__all__ = ["artifact_reference", "artifact_reference_parts", "is_artifact_reference"]
