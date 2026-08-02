"""Per-attempt correction artifact sinks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from receipt_intelligence.extraction.contracts.common import StageArtifact


class CorrectionArtifactSink(Protocol):
    @property
    def artifacts(self) -> tuple[StageArtifact, ...]: ...

    def write_json(self, name: str, payload: Any) -> None: ...


class NullCorrectionArtifactSink:
    @property
    def artifacts(self) -> tuple[StageArtifact, ...]:
        return ()

    def write_json(self, name: str, payload: Any) -> None:
        del name, payload


class FilesystemCorrectionArtifactSink:
    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._artifacts: list[StageArtifact] = []

    @property
    def artifacts(self) -> tuple[StageArtifact, ...]:
        return tuple(self._artifacts)

    def write_json(self, name: str, payload: Any) -> None:
        path = self._directory / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._artifacts.append(StageArtifact(name=path.stem, path=path))


__all__ = [
    "CorrectionArtifactSink",
    "FilesystemCorrectionArtifactSink",
    "NullCorrectionArtifactSink",
]
