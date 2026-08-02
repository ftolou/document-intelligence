"""Shared immutable values used at extraction stage boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

JsonObject: TypeAlias = dict[str, Any]
ReadonlyJsonObject: TypeAlias = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StageArtifact:
    name: str
    path: Path
    media_type: str = "application/json"

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        media_type = str(self.media_type or "").strip()
        if not name:
            raise ValueError("StageArtifact.name must not be empty.")
        if not media_type:
            raise ValueError("StageArtifact.media_type must not be empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "media_type", media_type)


__all__ = ["JsonObject", "ReadonlyJsonObject", "StageArtifact"]
