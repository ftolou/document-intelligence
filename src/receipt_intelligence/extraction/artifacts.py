"""Artifact naming and persistence helpers for extraction orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_alias(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copyfile(source, destination)


def build_artifact_paths(result_dir: Path, run_id: str) -> dict[str, Path]:
    return {
        "stage_trace": result_dir / f"{run_id}_extraction_stage_trace.json",
        "extraction_metrics": result_dir / f"{run_id}_extraction_metrics.json",
    }


__all__ = ["build_artifact_paths", "copy_alias", "save_json"]
